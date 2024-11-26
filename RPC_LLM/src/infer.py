import torch
import argparse
import torch.distributed.rpc as rpc
import time
from transformers import LlamaTokenizer
import os
from rpc_llama import LlamaForCausalLM

# 准备输入
prompts = [
    "The capital of France is",
    "The author of the Harry Potter books is",
    "The largest planet in our solar system is"
]


def inference(model, tokenizer):
    special_token = {'pad_token': '#'}
    tokenizer.add_special_tokens(special_token)
    # 批量编码
    batch = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    # 生成输出
    with torch.no_grad():
        st = time.time()
        outputs = model.generate(
            **batch,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        end = time.time()

    # 解码输出
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, output_text in enumerate(output_texts):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output_text}\n")
    print(f'Inference time: {end - st:.2f} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model-path', type=str, default="../../Models/Llama-2-7b-chat-hf",
                        help='The repo for the Llama2')
    parser.add_argument('--n-predict', type=int, default=128, help='Max tokens to predict')
    parser.add_argument("--gpu", action='store_true', default=False, help="whether to use gpu to accelerate")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.rank == -1:
        # 测试原始Llama模型的推理时延
        from transformers import LlamaForCausalLM

        # Load model
        model = LlamaForCausalLM.from_pretrained(args.model_path, device_map='auto')
        # 查看所使用的注意力机制
        print(f"model's attention type is {model.config._attn_implementation}")
        # for name, param in model.named_parameters():
        #     if param.device.type == "cpu":
        #         device_info = "CPU"
        #     else:
        #         device_info = f"GPU:{param.device.index}"
        #     print(f"Parameter: {name}, Device: {device_info} \n")
        #     print(f"Parameter: {name}, Shape: {param.shape}, Type: {param.dtype} \n")

        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        inference(model, tokenizer)

    elif args.rank == 0:
        print("work0 process init!\n")
        rpc.init_rpc("worker0",
                     rank=args.rank,
                     world_size=args.world_size,
                     backend=rpc.BackendType.TENSORPIPE,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                         rpc_timeout=6000,
                         _transports=["uv"],
                         _channels=["basic"]
                     )
                     )
        print("work0 process init finished!\n")

        # Load model
        model = LlamaForCausalLM.from_pretrained(args.model_path, state_dict=None)

        # 手动加载lm_head和model.norm的权重
        full_path = os.path.join(args.model_path, "pytorch_model-00003-of-00003.bin")
        weight_dic = torch.load(full_path)
        full_path = os.path.join(args.model_path, "pytorch_model-00001-of-00003.bin")
        weight_dic.update(torch.load(full_path))

        new_dic = {"model.norm.weight": weight_dic["model.norm.weight"], "lm_head.weight": weight_dic["lm_head.weight"],
                   "model.embed_tokens.weight": weight_dic["model.embed_tokens.weight"]}

        model.load_state_dict(new_dic)

        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name}, Shape: {param.shape}, Type: {param.device.type} , Para: {param}\n")

        remote_model_path = "../../Models/Llama-2-7b-chat-hf"
        model.init_decoders(remote_model_path)
        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        inference(model, tokenizer)
        model.print_time()
        rpc.shutdown()

    elif args.rank == 1:
        os.environ['MASTER_ADDR'] = '10.26.34.15'
        os.environ['MASTER_PORT'] = '8889'

        print("work1 process init!\n")
        rpc.init_rpc("worker1",
                     rank=args.rank,
                     world_size=args.world_size,
                     backend=rpc.BackendType.TENSORPIPE,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                         rpc_timeout=6000,
                         _transports=["uv"],
                         _channels=["basic"]
                     )
                     )
        print("work1 process init finished!\n")
        rpc.shutdown()
