import argparse
import math
import os
import time
import nltk
from emo import EMOLoss
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer, get_polynomial_decay_schedule_with_warmup, Trainer, AutoTokenizer
import torch
import datasets
import csv
import pdb
import logging
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dp_noise import add_noise
from utils import get_logger, LlamaLossFunc, LlamaEmbedding, LlamaEmbeddingConfig, lConfig, DpConfig
from data_helper import DataHelper, LlamaDataset, collate_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from modeling_llama import LlamaForCausalLM
from peft import get_peft_model, PeftModel, LoraConfig
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from picture import show_logits
from rouge import Rouge
from datasets import load_metric
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

log_name = "./Log/emo_lora4qkvo_Lap"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def mean(item: list) -> float:
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


metric = load_metric("rouge.py", trust_remote_code=True)


def compute_metrics(predictions, labels, tokenizer):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    # 获取数组的形状
    # 获取行数和列数
    rows = len(labels)
    cols = len(labels[0]) if rows > 0 else 0
    # 手动替换 -100 为 0
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] == -100:
                labels[i][j] = 0

    # pdb.set_trace()
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    # 对解码后的预测结果进行句子划分,并在每个句子之间插入换行符。这是因为 ROUGE 指标的计算需要按句子分隔。
    # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    # pdb.set_trace()
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    # 将所有指标值四舍五入到小数点后4位, 并返回结果字典。
    return {k: round(v, 4) for k, v in result.items()}


def accuracy(pred_ys, true_ys, masks):
    total = 0
    corr = 0

    for pred_y, true_y, mask in zip(pred_ys, true_ys, masks):
        pred_y = pred_y[:-1]
        true_y = true_y[1:]
        mask = mask[:-1]

        for p, t, m in zip(pred_y, true_y, mask):
            if m == 1:
                total += 1
                if p == t:
                    corr += 1

    return corr / total if total > 0 else 0


class Trainer:
    def __init__(self):
        self.testdata = None
        self.low_freq_words = None
        self.model = None
        self.valdata = []
        self.traindata = []
        self.device = torch.device("cuda:1")
        self.model_order = 0
        self.model_path_list = ["../../Models/Llama-2-7b-chat-hf",
                                "../../Models/Llama-3-8B-Instruct-hf"]
        self.model_path = self.model_path_list[self.model_order]
        self.data_order = 1
        self.datasets = ["../../Datasets/MedQA/questions/US/4_options/phrases_no_exclude_train.jsonl",
                         "../../Datasets/DialogSum/dialogsum.train.jsonl",
                         "../../Datasets/SST-2/train.jsonl"]

        self.test_datasets = ["../../Datasets/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",
                              "../../Datasets/DialogSum/dialogsum.test.jsonl",
                              "../../Datasets/SST-2/test.jsonl"]

        self.dataset_path = self.datasets[self.data_order]
        self.test_data_path = self.test_datasets[self.data_order]

        self.val_size = 500
        self.max_val_acc = -100.0

        self.save_dir = '../../Models_with_DP/Llama2-7b'
        self.batch_size = 64
        self.learning_rate = 3e-5
        self.end_learning_rate = 3e-6  # 3e-7
        self.gradient_accumulation_steps = 2
        self.epochs = 3
        self.log_every = self.gradient_accumulation_steps

        self.weight_decay = 0.1  # L2正则化系数
        self.lora = True
        if self.model_order == 1:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)

        self.model_pretrain = False

        print("Setup Data")
        self.get_data_loader()

        self.num_train_steps = (len(self.traindata) * self.epochs)
        self.warmup_steps = self.num_train_steps // 10
        self.eval_step = self.num_train_steps // 3
        self.word_freq = []

    def load_lora_weight(self):
        # 加载Lora权重
        self.model = PeftModel.from_pretrained(self.model, self.lora_path)

    def merge_lora_weight(self):
        # 合并Lora权重到主干模型
        self.model = self.model.merge_and_unload()

    def save_mergedModelWeight(self):
        # 保存合并后的模型权重
        self.model.save_pretrained(self.save_dir)
        print(f"model weight has been saved in {self.save_dir}")

    def get_data_loader(self):
        # 加载数据集
        data_obj = DataHelper(self.dataset_path, self.val_size)
        train_data, valid_data = data_obj.gen_data()

        valid_data = valid_data[:300]

        logger.info("train data size: {}".format(len(train_data)))
        logger.info("valid data size: {}".format(len(valid_data)))

        train_data_set = LlamaDataset(self.tokenizer, train_data, self.data_order)
        valid_data_set = LlamaDataset(self.tokenizer, valid_data, self.data_order)

        # 计算低频词
        # train_data_set.count_and_save_low_freq_words()

        self.traindata = DataLoader(train_data_set, batch_size=self.batch_size, drop_last=True, num_workers=6,
                                    shuffle=True, collate_fn=collate_fn)
        self.valdata = DataLoader(valid_data_set, batch_size=self.batch_size, drop_last=True, num_workers=6,
                                  shuffle=True, collate_fn=collate_fn)

    def get_test_data_loader(self):
        # 加载数据集
        data_obj = DataHelper(self.test_data_path, 3)
        test_data, _ = data_obj.gen_data()
        logger.info("test data size: {}".format(len(test_data)))
        test_data_set = LlamaDataset(self.tokenizer, test_data)
        self.testdata = DataLoader(test_data_set, batch_size=self.batch_size, drop_last=True, num_workers=6,
                                   shuffle=True, collate_fn=collate_fn)

    def LoadModelWeight(self):
        # Load model torch_dtype=torch.float16
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.pad_token_id = 0
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        if self.lora:
            lora_config = LoraConfig(
                r=lConfig.lora_r,
                lora_alpha=lConfig.lora_alpha,
                lora_dropout=lConfig.lora_dropout,
                target_modules=lConfig.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM"  # "CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

    def showTsne(self):
        for batch_data in tqdm(self.traindata):
            input_ids = batch_data[0].cuda()
            attention_mask = batch_data[1].cuda()
            labels = batch_data[2].cuda()
            noise_factors = batch_data[3].cuda()
            DpConfig.add_noise = False
            embedding = self.model.base_model.model.model.embed_tokens(input_ids=input_ids,
                                                                       attention_mask=attention_mask, labels=labels,
                                                                       noise_factors=noise_factors)
            DpConfig.add_noise = True
            noise_embedding = self.model.base_model.model.model.embed_tokens(input_ids=input_ids,
                                                                             attention_mask=attention_mask,
                                                                             labels=labels,
                                                                             noise_factors=noise_factors)

            combined_embedding = np.concatenate((embedding, noise_embedding), axis=0)

            # 对拼接后的embedding进行t-SNE降维
            tsne = TSNE(n_components=2, random_state=0)
            reduced_embedding = tsne.fit_transform(combined_embedding)

            reduced_raw = reduced_embedding[:len(embedding), :]
            reduced_noise = reduced_embedding[len(embedding):, :]

            # 绘制散点图
            plt.figure(figsize=(8, 8))
            plt.scatter(reduced_raw[:, 0], reduced_raw[:, 1],
                        color='blue', label='Raw Embedding', alpha=0.5)
            plt.scatter(reduced_noise[:, 0], reduced_noise[:, 1],
                        color='red', label='Encoded Embedding', alpha=0.5)

            plt.legend()
            plt.title('t-SNE Visualization of Raw and Encoded Embeddings')
            plt.show()

    def train(self):
        # Train
        logger.info("Start training")
        set_seed(42)
        step = 1
        train_word_preds = []
        train_word_labels = []
        train_masks = []
        train_losses = []
        logger.info(f"eval_step is {self.eval_step}")
        logger.info(f"warmup_step is {self.warmup_steps}")
        logger.info(f"num_train_steps is {self.num_train_steps}")

        self.model.train()

        start = time.time()
        for epoch in range(self.epochs):
            logger.info("----- Epoch {}/{} -----".format(epoch + 1, self.epochs))
            self.opt.zero_grad()
            for batch_data in tqdm(self.traindata):
                input_ids = batch_data[0].to(self.device)
                attention_mask = batch_data[1].to(self.device)
                labels = batch_data[2].to(self.device)
                prompt_mask = batch_data[3].to(self.device)

                noise_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                          prompt_masks=prompt_mask)
                noise_logits = noise_output["logits"]

                # DpConfig.emb_add_noise = False
                # clean_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # clean_logits = clean_output["logits"]
                # show_logits(noise_logits, clean_logits, labels)
                cost_em = self.model.base_model.model.lm_head.weight.to(noise_logits.device)

                loss = EMOLoss(noise_logits, labels, cost_em.float())

                if torch.isnan(loss):
                    pdb.set_trace()

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                predictions = torch.argmax(noise_logits, dim=-1)
                train_losses.append(loss)
                train_word_preds.extend(predictions.tolist())
                train_word_labels.extend(labels.tolist())
                train_masks.extend(attention_mask.tolist())

                if step % self.gradient_accumulation_steps == 0:
                    self.opt.step()
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scheduler.step()
                    self.opt.zero_grad()
                if step % self.log_every == 0 and step:
                    acc = accuracy(pred_ys=train_word_preds, true_ys=train_word_labels, masks=train_masks)
                    result = compute_metrics(train_word_preds, train_word_labels, self.tokenizer)
                    logger.info(
                        f"train step: {step:4d}, "
                        f"loss: {mean(train_losses):.4f}, "
                        f"rouge1: {result['rouge1']}, "
                        f"rougeL: {result['rougeL']}, "
                        f"acc: {acc:.4f}"
                    )
                    train_losses = []
                    train_word_preds = []
                    train_word_labels = []
                    train_masks = []

                if step % self.eval_step == 0 and step:
                    self.eval(step)
                    self.model.train()
                    self.model.config.use_cache = False
                step = step + 1
        # 评估
        self.eval(step)
        end = time.time()
        logger.info(f"total train time: {(end - start) / 3600:.4f} h!")

    def eval(self, step):
        self.model.eval()
        self.model.config.use_cache = True

        with (torch.no_grad()):
            eval_losses = []
            eval_word_preds = []
            eval_word_labels = []
            eval_masks = []
            eval_rouge1 = []
            eval_rougel = []

            for batch_data in tqdm(self.valdata):
                input_ids = batch_data[0].to(self.device)
                attention_mask = batch_data[1].to(self.device)
                labels = batch_data[2].to(self.device)
                prompt_mask = batch_data[3].to(self.device)

                noise_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                          prompt_masks=prompt_mask)
                noise_logits = noise_output["logits"]

                # DpConfig.add_noise = False

                cost_em = self.model.base_model.model.lm_head.weight.to(noise_logits.device)
                # cost_em = self.model.base_model.model.model.embed_tokens.weight.to(noise_logits.device)

                eval_loss = EMOLoss(noise_logits, labels, cost_em.float())
                # self.lossfun(pred_y=noise_logits, true_y=labels, cost_embedding=cost_em)

                eval_predictions = torch.argmax(noise_logits, dim=-1)
                eval_losses.append(eval_loss)

                if torch.isnan(eval_loss):
                    pdb.set_trace()
                # logger.info(f"\nloss: {eval_loss:.2f}")
                eval_word_preds.extend(eval_predictions.tolist())
                eval_word_labels.extend(labels.tolist())
                eval_masks.extend(attention_mask.tolist())
                result = compute_metrics(eval_word_preds, eval_word_labels, self.tokenizer)
                logger.info(
                    f"eval  step: {step:4d}, "
                    f"rouge1: {result['rouge1']}, "
                    f"rougeL: {result['rougeL']}, "
                )
                eval_rouge1.append(result['rouge1'])
                eval_rougel.append(result['rougeL'])

            acc = accuracy(pred_ys=eval_word_preds, true_ys=eval_word_labels, masks=eval_masks)
            # mauves = cal_mauve(eval_word_preds, eval_word_labels, masks=eval_masks)
            logger.info(
                f"eval step: {step:4d}, "
                f"loss: {mean(eval_losses):.4f}, "
                f"rouge1: {mean(eval_rouge1):.2f}, "
                f"rougeL: {mean(eval_rougel):.2f}, "
                f"acc: {acc:.4f}"
            )
            map = {0: "MedQA", 1: "DialogSum", 2: "SST-2"}
            mid_name = (map[self.data_order]) + "_" + str(DpConfig.lap_epsilon)
            lora_model_path = os.path.join(self.save_dir, mid_name, str(step))

            if acc >= self.max_val_acc:
                self.max_val_acc = acc
                if self.lora:
                    if not os.path.exists(lora_model_path):
                        os.makedirs(lora_model_path)
                    self.model.save_pretrained(lora_model_path)
                logger.info(f"max val acc is {self.max_val_acc}")
                logger.info(f"step {step} Lora model has been saved {lora_model_path}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    trainer = Trainer()
    log_name = log_name + "_M" + str(trainer.model_order + 2) + "_D" + str(trainer.data_order) + "_N" + str(
        DpConfig.lap_epsilon) + ".txt"
    logger = get_logger("llama_dp", log_name)

    if args.mode == 0:
        print("Setup Model")
        trainer.LoadModelWeight()
        if trainer.lora:
            trainer.model.print_trainable_parameters()

        print("Setup optimizer")
        trainer.opt = torch.optim.AdamW(trainer.model.parameters(), lr=trainer.learning_rate, betas=(0.9, 0.95),
                                        eps=1e-5, weight_decay=trainer.weight_decay)
        trainer.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=trainer.opt,
                                                                      num_warmup_steps=trainer.warmup_steps,
                                                                      num_training_steps=trainer.num_train_steps,
                                                                      lr_end=trainer.end_learning_rate)
        trainer.train()

    elif args.mode == 1:
        # 测试模型
        trainer.lora = False
        print("Setup Model")
        trainer.LoadModelWeight()
        print("loading lora weight!")
        trainer.load_lora_weight()

        trainer.eval(1)

    elif args.mode == 2:
        # 画图
        print("Setup Model")
        # trainer.LoadModelWeight()
        # trainer.showTsne()

    else:
        print("error!\n")
