import os
import sys
import argparse
import torch
import transformers
import torch.nn as nn
from datetime import datetime
from typing import Literal, Iterable
import logging
import pprint

import transformers.models
import tqdm
from transformers import PreTrainedModel
from functools import partial

from mxmoe.quant.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, get_hadK, matmul_hadU_cuda
from mxmoe.quant.moe_utils import (
    get_expert_linears, get_linears_in_one_expert, get_attn_linears,
    get_moe_gate_linears,
)

LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
QWEN2_MOE_MODEL = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeForCausalLM
MIXTRAL_MODEL = transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM
DS2_MODEL = "deepseek_v2"

ROTATE_SUPPORT_MODEL = [LLAMA_MODEL, QWEN2_MOE_MODEL, MIXTRAL_MODEL, "deepseek_v2"]

def get_model_type(model: PreTrainedModel):
    if model.config.model_type=="deepseek_v2":
        return DS2_MODEL
    for t in ROTATE_SUPPORT_MODEL[:-1]:
        if isinstance(model, t):
            return t
    else:
        raise ValueError(f"Unknown model type {model}")

# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace) and name not in ["kv_a_layernorm"]:
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        # Fusion already absorbed the original scale. Keep a native-shaped unit
        # weight so save/reload into stock RMSNorm is an identity multiply.
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.to(dtype=x.dtype, device=x.device)).to(input_dtype)


class NewQwen2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight.to(dtype=hidden_states.dtype, device=hidden_states.device)).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.no_grad()
def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    # Calculating new weight and bias
    for linear in linear_layers:
        if isinstance(linear, nn.Linear):
            linear_dev = linear.weight.device
            linear_dtype = linear.weight.dtype
            W_ = linear.weight.data.double()
            linear.weight.data = (W_ * layernorm.weight.double().to(linear_dev)).to(linear_dtype)
        else:
            linear_dev = linear.data.device
            linear_dtype = linear.data.dtype
            W_ = linear.data.double()
            linear.data = (W_ * layernorm.weight.double().to(linear_dev)).to(linear_dtype)

        # if hasattr(layernorm, 'bias'):
        #     if linear.bias is None:
        #         linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
        #     linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
        #     linear.bias.data = linear.bias.data.to(linear_dtype)


@torch.no_grad()
def fuse_layer_norms(model):

    rmsnorm_map = {
        LLAMA_MODEL: (transformers.models.llama.modeling_llama.LlamaRMSNorm, RMSN),
        QWEN2_MOE_MODEL: (transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm, NewQwen2MoeRMSNorm),
        MIXTRAL_MODEL: (transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm, NewQwen2MoeRMSNorm),
        DS2_MODEL: (type(model.model.norm), NewQwen2MoeRMSNorm),
    }

    model_type = get_model_type(model)

    assert model_type in ROTATE_SUPPORT_MODEL, f"Unsupported model {model}"

    def get_embeding_layer():
        if model_type in [LLAMA_MODEL, QWEN2_MOE_MODEL, MIXTRAL_MODEL, DS2_MODEL]:
            return model.model.embed_tokens
        else:
            raise ValueError(f'Unknown model type {model_type}')
    def get_transformer_layers():
        if model_type in [LLAMA_MODEL, QWEN2_MOE_MODEL, MIXTRAL_MODEL, DS2_MODEL]:
            return model.model.layers
    def get_pre_head_norm():
        if model_type in [LLAMA_MODEL, QWEN2_MOE_MODEL, MIXTRAL_MODEL, DS2_MODEL]:
            return model.model.norm
        else:
            raise ValueError(f'Unknown model type {model_type}')
    def get_lm_head():
        if model_type in [LLAMA_MODEL, QWEN2_MOE_MODEL, MIXTRAL_MODEL, DS2_MODEL]:
            return model.lm_head
        else:
            raise ValueError(f'Unknown model type {model_type}')

    # Embedding fusion
    W = get_embeding_layer()
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers()
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer_idx, layer in enumerate(layers):
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type in [QWEN2_MOE_MODEL, MIXTRAL_MODEL, DS2_MODEL]:
            if model_type == DS2_MODEL:
                qkvs = [layer.self_attn.q_proj, layer.self_attn.kv_a_proj_with_mqa]
            else:
                attn_linears = get_attn_linears(model, layer_idx)
                qkvs = [attn_linears['q_proj'], attn_linears['k_proj'], attn_linears['v_proj']]
            expert_blocks = get_expert_linears(model, layer_idx, exclude_non_moe_layer=False)        
            gate_ups = []
            moe_gates = get_moe_gate_linears(model, layer_idx)
            for expert in expert_blocks:
                linears = get_linears_in_one_expert(expert)
                gate_ups.extend([linears['gate'], linears['up']])

            fuse_ln_linear(layer.post_attention_layernorm, gate_ups)
            fuse_ln_linear(layer.post_attention_layernorm, moe_gates)
            fuse_ln_linear(layer.input_layernorm, qkvs)
        else:
            raise ValueError(f'Unknown model type {model_type}')

    fuse_ln_linear(get_pre_head_norm(), [get_lm_head()])
    
    old_rmsnorm = rmsnorm_map[model_type][0]
    new_rmsnorm = rmsnorm_map[model_type][1]
    replace_modules(
        model,
        old_rmsnorm,
        lambda _: new_rmsnorm(model.config.hidden_size),
        replace_layers=False,
    )

    
def random_orthogonal_matrix(size):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode: Literal["random", "hadamard"]):
    if mode == 'random':
        return random_orthogonal_matrix(size)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size)
    else:
        raise ValueError(f'Unknown mode {mode}')


class ModelRotator:
    '''
    In general, Y = X @ W.T
    As a result, Y = (X@Q) @ (W@Q).T = X @ W.T
    '''
    def __init__(
        self,
        model,
        rotation_mode: Literal["random", "hadamard"],
        dev=torch.device("cuda:0"),
    ):
        # self.model = model
        self.hidden_size: int = model.config.hidden_size
        # self.moe_hidden_size = model.config.moe_hidden_size
        self.model_type = get_model_type(model)
        print(self.model_type)

        self.rotation_mode = rotation_mode
        self.work_dev = dev
        
        assert rotation_mode in ["random", "hadamard"], f"Unknown rotation mode {rotation_mode}"
        assert self.model_type in ROTATE_SUPPORT_MODEL, f"Unsupported model {model}"

        self.mlp_inputs:   list[list[nn.Linear]] = []  # gate, up, [moe_gate_linears]
        self.mlp_outputs:  list[list[nn.Linear]] = []  # down
        self.attn_inputs:  list[list[nn.Linear]] = []  # q, k, v
        self.attn_outputs: list[list[nn.Linear]] = []  # o
        self.lm_head:   nn.Linear = model.lm_head
        self.embedding: nn.Embedding = model.model.embed_tokens
        self.num_layers = len(model.model.layers)

        for layer_idx, layer in enumerate(model.model.layers):
            if self.model_type in [QWEN2_MOE_MODEL, MIXTRAL_MODEL]:
                attn_linears = get_attn_linears(model, layer_idx)
                expert_blocks = get_expert_linears(model, layer_idx, exclude_non_moe_layer=False)        

                gate_ups = get_moe_gate_linears(model, layer_idx)
                downs = []
                qkv = [attn_linears['q_proj'], attn_linears['k_proj'], attn_linears['v_proj']]
                o = [attn_linears['o_proj']]

                for expert in expert_blocks:
                    linears = get_linears_in_one_expert(expert)
                    gate_ups.extend([linears["gate"], linears["up"]])
                    downs.extend([linears["down"]])
            elif self.model_type in [DS2_MODEL]:
                # kv_a_proj_with_mqa
                # q_proj
                qkv = [layer.self_attn.q_proj, layer.self_attn.kv_a_proj_with_mqa]
                o   = [layer.self_attn.o_proj]

                # attn_linears = get_attn_linears(model, layer_idx)
                expert_blocks = get_expert_linears(model, layer_idx, exclude_non_moe_layer=False)        

                gate_ups = get_moe_gate_linears(model, layer_idx)
                downs = []

                for expert in expert_blocks:
                    linears = get_linears_in_one_expert(expert)
                    gate_ups.extend([linears["gate"], linears["up"]])
                    downs.extend([linears["down"]])
            elif self.model_type in [LLAMA_MODEL]:
                gate_ups = [layer.mlp.gate_proj, layer.mlp.up_proj]
                downs = [layer.mlp.down_proj]
                qkv = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
                o = [layer.self_attn.o_proj]

            self.mlp_inputs.append(gate_ups)
            self.mlp_outputs.append(downs)
            self.attn_inputs.append(qkv)
            self.attn_outputs.append(o)


    @torch.no_grad()
    def rotate_attention_inputs(self, layer_idx:int, Q: torch.Tensor) -> None:
        for W in self.attn_inputs[layer_idx]:
            dtype = W.weight.dtype
            W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
            W.weight.data.copy_(torch.matmul(W_, Q).to(dtype=dtype))


    @torch.no_grad()
    def rotate_attention_output(self, layer_idx:int, Q: torch.Tensor) -> None:
        for W in self.attn_outputs[layer_idx]:
            dtype = W.weight.dtype
            W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
            W.weight.data.copy_(torch.matmul(Q.T, W_).to(dtype=dtype))


    @torch.no_grad()
    def rotate_mlp_input(self, layer_idx: int, Q: torch.Tensor) -> None:
        for W in self.mlp_inputs[layer_idx]:
            if isinstance(W, nn.Linear):
                dtype = W.weight.dtype
                W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
                W.weight.data.copy_(torch.matmul(W_, Q).to(dtype=dtype))
            else:
                dtype = W.data.dtype
                W_ = W.data.to(device=self.work_dev, dtype=torch.float64)
                W.data.copy_(torch.matmul(W_, Q).to(dtype=dtype))
    

    @torch.no_grad()
    def rotate_mlp_output(self, layer_idx:int, Q: torch.Tensor) -> None:
        for W in self.mlp_outputs[layer_idx]:
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
            W.weight.data.copy_(torch.matmul(Q.T, W_).to(dtype=dtype))
            # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output


    @torch.no_grad()
    def rotate_head(self, Q: torch.Tensor) -> None:
        # Rotate the head.
        W = self.lm_head
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
        W.weight.data.copy_(torch.matmul(W_, Q).to(dtype=dtype))


    @torch.no_grad()
    def rotate_embedding(self, Q: torch.Tensor) -> None:
        # Rotate the embeddings.
        W = self.embedding
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=self.work_dev, dtype=torch.float64)
        W.weight.data.copy_(torch.matmul(W_, Q).to(dtype=dtype))


    # def rotate_ov_proj(self, layer, model_type, head_num, head_dim) -> None:
    #     v_proj = layer.self_attn.v_proj
    #     if model_type == model_utils.LLAMA_MODEL:
    #         o_proj = layer.self_attn.o_proj
    #     elif model_type == model_utils.OPT_MODEL:
    #         o_proj = layer.self_attn.out_proj
    #     else:
    #         raise ValueError(f'Unknown model type {model_type}')
    #     apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    #     apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

    def online_had_down_proj(self,layer_idx: int, plug_hook_only:bool=False) -> list[torch.utils.hooks.RemovableHandle]:
        handles = []
        for W in self.mlp_outputs[layer_idx]:
            if not plug_hook_only:
                # 1. reverse_online_had_down_proj
                apply_exact_had_to_linear(W, had_dim=-1, output=False)

            # 2. online had hook
            had_K, K = get_hadK(W.in_features)

            def online_had(m: nn.Module, inp, had: torch.Tensor, K: int):
                x: torch.Tensor = inp[0]
                if x.shape[0] == 0: return x
                x_dtype = x.dtype
                return matmul_hadU_cuda(x, had.to(x.device), K).to(x_dtype)

            hook = partial(online_had, had=had_K, K=K)
            handles.append(W.register_forward_pre_hook(hook))
        return handles
    

    def online_had_o_proj(self, layer_idx: int, plug_hook_only=False) -> list[torch.utils.hooks.RemovableHandle]:
        return []


    def plug_online_had_hook(self) -> list[torch.utils.hooks.RemovableHandle]:
        online_had_hooks = []
        for idx in range(self.num_layers):
            online_had_hooks.extend(self.online_had_down_proj(idx, plug_hook_only=True))
            online_had_hooks.extend(self.online_had_o_proj(idx, plug_hook_only=True))
        return online_had_hooks

    @torch.no_grad()
    def rotate_model(self, model, enable_online_rotation: bool = False) -> list[torch.utils.hooks.RemovableHandle]:
        ori_dev = model.device
        # 1. Fuse the rms-norm weight into the linear layers
        fuse_layer_norms(model)

        # 2. Rotate the model
        Q = get_orthogonal_matrix(self.hidden_size, self.rotation_mode).to(self.work_dev)

        self.rotate_embedding(Q)
        self.rotate_head(Q)

        online_had_hooks = []
        for idx in tqdm.tqdm(range(self.num_layers), unit="layer", desc="Rotating"):
            if "cpu" in ori_dev.type:
                model.model.layers[idx] = model.model.layers[idx].to(self.work_dev)

            self.rotate_attention_inputs(idx, Q)
            self.rotate_attention_output(idx, Q)
            self.rotate_mlp_input(idx, Q)
            self.rotate_mlp_output(idx, Q)

            if enable_online_rotation:
                online_had_hooks.extend(self.online_had_down_proj(idx))
                online_had_hooks.extend(self.online_had_o_proj(idx))
            # apply_hadamard(layer_adapter.get_mlp_output())
            # apply_hadamard_headwise(layer_adapter.get_v_proj(), head_dim)
            # apply_hadamard(layer_adapter.get_attention_output(), head_dim)

            if "cpu" in ori_dev.type:
                model.model.layers[idx] = model.model.layers[idx].to(ori_dev)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return online_had_hooks
