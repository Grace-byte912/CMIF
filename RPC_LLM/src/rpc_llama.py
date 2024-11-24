import os
import time
import warnings
import torch.nn.functional as F
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from typing import List, Optional, Tuple, Union
import torch.nn as nn
from torch.distributed import rpc
import torch
from torch.distributed.rpc import RRef
from transformers.models.llama.modeling_llama import LLAMA_START_DOCSTRING, logger, \
    LLAMA_INPUTS_DOCSTRING, LlamaMLP, LLAMA_ATTENTION_CLASSES
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers import LlamaPreTrainedModel, add_start_docstrings, LlamaConfig
from collections.abc import Iterable

'''
遍历模块的所有参数。
为每个参数创建一个 RRef，这使得参数可以在远程节点上被引用和操作。
返回包含所有参数 RRef 的列表。
'''


def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


'''
参数:
method: 要执行的函数
rref: 包含目标对象的RRef。
*args 和 **kwargs: 要传递给方法的参数。
功能:
使用 rref.local_value() 获取RRef所引用的本地对象。
在该本地对象上调用指定的方法，传入相应的参数。
返回方法的执行结果。
'''


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


'''
参数:
method: 要在远程对象上调用的方法。 
rref: 指向远程对象的RRef。
*args 和 **kwargs: 被传递给远程方法的参数。
功能:
使用 rpc.rpc_sync 同步地调用远程函数_call_method，并将计算结果返回。
这里的 rpc.rpc_sync 调用确保了方法在远程节点上执行，并等待执行结果。
'''


# 调用远程节点上的对象的给定方法，阻塞执行
def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args, kwargs=kwargs, timeout=0)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # print(f"self.weight is on device: {self.weight.device}")
        hidden_states = hidden_states.to(self.weight.device)

        # self.to(hidden_states.device)
        # print(f"### self.weight is on device: {self.weight.device}")
        # print(f"### hidden_states is on device: {hidden_states.device}")

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaDecoderLayers(nn.Module):
    def __init__(self, config: LlamaConfig):
        super(LlamaDecoderLayers, self).__init__()
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        if torch.cuda.is_available():
            for idx, layer in enumerate(self.layers):
                if idx <= 14:
                    layer.to("cuda:0")
                else:
                    layer.to("cuda:1")
        self.forward_time = 0.0

    def get_time(self):
        return self.forward_time

    def __getitem__(self, index):
        return self.layers[index]

    def get_past_key_values(self):
        past_key_values = getattr(getattr(self.layers[0], "self_attn", {}), "past_key_value", None)
        if past_key_values is not None:
            past_key_values = past_key_values.cpu()
        return past_key_values

    def get_flag(self):
        return hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value")

    def load_weights(self, pretrained_model_name_or_path):
        bin_files = [f for f in os.listdir(pretrained_model_name_or_path) if f.endswith(".bin")]
        state_dict = {}
        for bin_file in tqdm(bin_files):
            bin_file_path = os.path.join(pretrained_model_name_or_path, bin_file)
            state_dict.update(torch.load(bin_file_path))

        print("The length of LlamaDecoderLayers parameters dict is ", len(state_dict.keys()))

        new_state_dict = {}
        for name, param in state_dict.items():
            if "model.layers." in name:
                # print(name)
                rename = name.replace("model.layers.", "")
                new_state_dict[rename] = param

        print("Loading LlamaDecoderLayers parameters!\n")
        self.layers.load_state_dict(new_state_dict)
        print("Loaded LlamaDecoderLayers parameters!\n")
        # for name, param in self.layers.named_parameters():
        #     print(f"Parameter {name} is on device: {param.device}")

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        # print("torch.cuda.is_available()", torch.cuda.is_available())

        def move_to_gpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                # 如果 obj 是字典，递归地移动每一个键值对
                return {k: move_to_gpu(v) for k, v in obj.items()}
            elif isinstance(obj, DynamicCache):
                obj.key_cache = [tensor.to(device) for tensor in obj.key_cache]
                obj.value_cache = [tensor.to(device) for tensor in obj.value_cache]
                return obj
            else:
                return obj

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        index = 0

        st = time.time()
        for decoder_layer in self.layers:
            # print("decoder_layer: ", index)
            if torch.cuda.is_available() and index <= 14:
                device = "cuda:0"
            elif torch.cuda.is_available():
                device = "cuda:1"
            hidden_states = hidden_states.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            position_ids = position_ids.to(device) if position_ids is not None else None
            if past_key_values is not None:
                past_key_values = move_to_gpu(past_key_values)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            # print("after decoding hidden_states: ", hidden_states)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            index = index + 1

        end = time.time()
        self.forward_time = self.forward_time + (end - st)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        # return all_hidden_states, hidden_states, all_self_attns, next_cache

        hidden_states = hidden_states.to("cpu")

        if all_hidden_states is not None:
            all_hidden_states = tuple(hs.to("cpu") for hs in all_hidden_states)

        if all_self_attns is not None:
            all_self_attns = tuple(sa.to("cpu") for sa in all_self_attns)

        def move_to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to("cpu")
            elif isinstance(obj, tuple):
                return tuple(move_to_cpu(o) for o in obj)
            else:
                return obj

        if next_cache is not None:
            next_cache = move_to_cpu(next_cache)

        return (
            all_hidden_states,
            hidden_states,
            all_self_attns,
            next_cache
        )


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.decoders = rpc.remote("worker1", LlamaDecoderLayers, (config,))
        # self.decoders = LlamaDecoderLayers(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.total_time = 0.0
        self.embed_time = 0.0

    def get_time(self):
        return self.total_time

    def get_embed_time(self):
        return self.embed_time

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        st = time.time()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # print("input_embeddings' shape:", inputs_embeds.shape)
        # print("input embeddings is :", inputs_embeds)
        end = time.time()
        self.embed_time = self.embed_time + end - st

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # all_hidden_states：用于存储每一层解码器的隐藏状态，如果output_hidden_states为True。
        # all_self_attns：用于存储每一层解码器的自注意力权重，如果output_attentions为True。
        # next_decoder_cache：用于存储解码器的缓存，用于加速后续的解码过程。
        # print("Type of inputs_embeds:", type(inputs_embeds))
        # print("inputs_embeds: ", inputs_embeds)
        # print("LlamaModel forward:", type(past_key_values))
        st = time.time()
        all_hidden_states, hidden_states, all_self_attns, next_cache = _remote_method(LlamaDecoderLayers.forward,
                                                                                      self.decoders,
                                                                                      inputs_embeds,
                                                                                      causal_mask,
                                                                                      position_ids,
                                                                                      past_key_values,
                                                                                      output_attentions,
                                                                                      use_cache,
                                                                                      output_hidden_states,
                                                                                      cache_position
                                                                                      )
        end = time.time()
        self.total_time += end - st

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            print("not return dict")
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        # print("hidden_states' shape:", hidden_states.shape)
        # print("hidden_states is", hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        flag = _remote_method(LlamaDecoderLayers.get_flag, self.decoders)

        if flag:  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask,
                                                                                            min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                : mask_shape[0], : mask_shape[1], offset: mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                    torch.jit.is_tracing()
                    or isinstance(input_tensor, torch.fx.Proxy)
                    or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        self.total_time = 0.0
        self.lm_head_time = 0.0

    def print_time(self):
        tmp = _remote_method(LlamaDecoderLayers.get_time, self.model.decoders)
        print("################################# Time Report ########################################")
        print(f"DecoderLayers forward time in another server: {tmp:.2f}")
        print(f"LlamaModel's DecoderLayers running time including remote call: {self.model.get_time():.2f}")
        tmp = self.model.get_time() - tmp
        print(f"transport time between two processes: {tmp:.2f}")
        print(f"LlamaModel's embedding forward time: {self.model.get_embed_time():.2f}")
        print(f"LlamaForCausalLM's model forward time: {self.get_time():.2f} ")
        print(f"LlamaForCausalLM's lm_head forward time: {self.lm_head_time:.2f}")

    def init_decoders(
            self,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    ):
        print("init decoders!\n")
        st = time.time()
        _remote_method(LlamaDecoderLayers.load_weights, self.model.decoders, pretrained_model_name_or_path)
        end = time.time()
        print("LlamaDecoderLayers weights loading time: ", end - st)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_time(self):
        return self.total_time

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="LlamaConfig")
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

       """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("LlamaForCausalLM forward:", type(past_key_values))
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        st = time.time()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        end = time.time()
        self.total_time += end - st
        hidden_states = outputs[0]
        # print("outputs shape ", hidden_states.shape)
        # print("output layer hidden_states is ", hidden_states)
        # print("shape", hidden_states.shape)

        st = time.time()
        # 如果使用了参数并行化
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        end = time.time()
        self.lm_head_time += end - st

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # print("logits' shape :", logits.shape)
        # print("logits' value :", logits)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None,
            **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        # 检查是否有静态缓存
        if past_key_values is None:
            past_key_values = _remote_method(LlamaDecoderLayers.get_past_key_values, self.model.decoders)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
