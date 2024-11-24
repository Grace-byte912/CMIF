import os
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.distributed.optim import DistributedOptimizer
import torch.nn as nn
import datasets
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)
from rpc_llama import LlamaForCausalLM
from transformers.trainer import TRAINING_ARGS_NAME
from peft import get_peft_model, LoraConfig, TaskType, PromptTuningConfig, PromptEncoderConfig, PrefixTuningConfig


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="/home/yuhonglan/llm/Finetune_LLAMA/Data_sample/UMLSE_Train_Tokenized")
    model_path: str = field(default="/home/yuhonglan/Models/Llama-2-7b-chat-hf")


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


@dataclass
class RpcArguments:
    rank: int
    world_size: int = field(default=2)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def run_trainer(finetune_args, peft_args, training_args):
    print("Setup tokenizer")
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)

    print("Setup Model")
    model = LlamaForCausalLM.from_pretrained(
        finetune_args.model_path,
        load_in_8bit=True,
        trust_remote_code=True
        # device_map='auto',
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # model.is_parallelizable = False
    # model.model_parallel = False
    # model.lm_head为最后的linear层，将该层的输出转换为浮点数类型
    # print(model.get_lm_head_para)
    # model.lm_head = CastOutputToFloat(model.get_lm_head_para)
    model.config.use_cache = False

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=peft_args)
    model = get_peft_model(model, peft_config)

    print("Setup distributed optimizer")
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for epoch in range(int(training_args.num_train_epochs)):
            print("Training epoch {}".format(epoch))
            # create distributed autograd context
            for data, target in get_next_batch(dataset, tokenizer):
                with dist_autograd.context() as context_id:
                    output, hidden = model(data)
                    loss = criterion(output, target)
                    # run distributed backward pass
                    dist_autograd.backward(context_id, [loss])
                    # run distributed optimizer
                    opt.step(context_id)
                    # not necessary to zero grads as each iteration creates a different
                    # distributed autograd context which hosts different grads

        # model.save_pretrained(training_args.output_dir)


def main():
    r"""
        A wrapper function that initializes RPC, calls the function, and shuts down
        RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'

    finetune_args, peft_args, training_args, rpc_args = HfArgumentParser(
        (FinetuneArguments, PEFTArguments, TrainingArguments, RpcArguments)
    ).parse_args_into_dataclasses()

    if rpc_args.rank == -1:
        # 在使用mp.spawn函数时，如果没有明确设置rank的值，它会自动为每个进程分配一个唯一的rank
        mp.spawn(rpc_args.run_worker,
                 args=(rpc_args.world_size, finetune_args, training_args),
                 nprocs=rpc_args.world_size,
                 join=True)
    elif rpc_args.rank == 0:
        print("work0 process init!\n")
        rpc.init_rpc("work0", rank=rpc_args.rank, world_size=rpc_args.world_size)
        print("work0 process init finished!\n")

        processes = []
        p = mp.Process(target=rpc_args.run_worker, args=(rpc_args.rank, rpc_args.world_size, finetune_args, training_args))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
    elif rpc_args.rank == 1:
        print("work1 process init!\n")
        rpc.init_rpc("work1", rank=rpc_args.rank, world_size=rpc_args.world_size)
        print("work1 process init finished!\n")

        run_trainer(finetune_args, peft_args, training_args)

        processes = []
        p = mp.Process(target=rpc_args.run_worker, args=(rpc_args.rank, rpc_args.world_size, finetune_args, training_args))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

    # block until all rpc-processes finish
    rpc.shutdown()


if __name__ == "__main__":
    main()
