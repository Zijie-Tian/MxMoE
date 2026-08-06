import os
import json
import time
import torch
import pickle
import argparse
import warnings
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from enum import StrEnum
from functools import partial
from tqdm import tqdm
from torch import Tensor
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from typing import Literal
from mxmoe.quant.evaluator import Evaluator

from mxmoe.quant.gptq import GPTQ
from mxmoe.quant.gptq import Quantizer as GPTQuantizer
from mxmoe.quant.data_utils import get_wikitext2
from mxmoe.quant.moe_utils import (
    get_expert_linears,
    get_linears_in_one_expert, get_linear_block_weight,
    recover_weight_from_cpu,
    substitue_moe_weights, offload_moe_weights,
    load_hf_model, build_max_memory, MOE_WEIGHT_NAME_MAP,
)
from mxmoe.kernels.qconfig import (
    QLinearConfig, QModelConfig, QLayerConfig,
    build_qmodel_cfg_from_json, get_all_wbits,
    build_uni_qexpert_cfg, build_uni_qlayer_cfg, build_uni_qmodel_cfg
)
from project_config import *
from logger_utils import init_logger, setup_logger

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

def quant_minmax(
    t: Tensor,
    bits: int|float,
    gsize: int,
    sym: bool,
    clip: tuple[float, float]=(1.0, 1.0),
    dim=-1
):
    """
    Dynamic Quantiztion
    Assume the last dim is reduction dim (group dim)

        bits: bitwidth of quantized tensor
        gsize: -1 means per-channel quantization for weight; per-token quantization for activation
        sym: True for symmetric quantization; False for asymmetric quantization
        clip: clip the min/max value to alleviate the outlier problem

    return: (quant_tensor, scale, zp), shape of quant param: [bs, gsize*N, 1]
    """
    if isinstance(bits, float): assert bits == 1.5, "use 1.5 to represent Terinary quantization"

    gsize = t.shape[dim] if gsize == -1 else gsize
    reshape_t = t.reshape(-1, gsize)
    if bits == 1.5:
        upper, lower, sym = 1.0, -1.0, True
    elif bits == 1:
        upper, lower, sym = 1.0, 0.0, False
    else:
        upper = (1 << (bits - 1)) - 1 if sym else (1 << bits) - 1
        lower = -upper if sym else 0

    if not sym:
        gmin, gmax = reshape_t.aminmax(dim=dim, keepdim=True)    
        gmin *= clip[0]
        gmax *= clip[1]
        scale = (gmax - gmin) / upper
        zp = gmin
    else:
        gmax = reshape_t.abs().amax(dim=dim, keepdim=True)
        scale = gmax / upper
        zp = 0
    
    quant_t = reshape_t.sub(zp).div(scale).clamp(lower, upper).round()
    
    return quant_t.reshape(t.shape), scale, zp


class Quantizer:
    def __init__(self, bits: int, sym: bool, gsize: int, clip:tuple[float, float]=(1.0, 1.0)):
        self.sym = sym
        self.bits = bits
        self.gsize = gsize
        self.clip = clip

    def quant(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.bits == 16: return (t, 1.0, 0.0)
        gsize = t.shape[-1] if self.gsize == -1 else self.gsize
        return quant_minmax(t, self.bits, gsize, self.sym, self.clip)

    def dequant(self, t: Tensor, scale: Tensor, zp: Tensor):
        if self.bits == 16: return t
        gsize = t.shape[-1] if self.gsize == -1 else self.gsize
        return t.reshape(-1, gsize).mul(scale).add(zp).reshape(t.shape)
    
    def fake_quant(self, t: Tensor) -> Tensor:
        if self.bits == 16: return t
        return self.dequant(*self.quant(t))

    def __repr__(self):
        return f'Quantizer(bits={self.bits}, sym={self.sym}, gsize={self.gsize}, clip={self.clip})'


class QMethod(StrEnum):
    # WxAy
    LLM_INT8 = "llm.int8"
    SMOOTH_QUANT = "smoothquant"
    WxAy_NAIVE = "wxay_naive"
    GPTQ_HAD = "gptq_had"
    RTN_HAD = "rtn_had"

    # WxA16
    GPTQ = "gptq"
    AWQ = "awq"
    WxA16_NAIVE = "wxa16_naive"


def mlp_inp_quant_hook(m: nn.Module, inp, quantizer: Quantizer):
    '''
    this hook will quantize the input of the MLP layer
    '''
    return quantizer.fake_quant(inp[0])

def smooth_act_quant_hook(m: nn.Module, inp:tuple, quantizer: Quantizer, smooth_scale: Tensor):
    return quantizer.fake_quant(inp[0].div(smooth_scale))

@torch.no_grad()
def prepare_inps(model: PreTrainedModel, dataloader: list[Tensor]):
    '''
    dataloader: list of input, length of each input is seqlen
    return:
        inps: [nsamples, seqlen, hidden_size]
        attention_mask
        position_ids
    '''
    dev = torch.device("cuda:0")

    sample_seq_len = dataloader[0].shape[-1]
    num_samples = len(dataloader)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    ori_dev = model.model.embed_tokens.weight.device

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.norm = model.model.norm.to(dev)
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    print(model.model.embed_tokens.weight.device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((num_samples, sample_seq_len, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(dev)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in tqdm(dataloader, desc="Preparing inputs"):
        try:
            # Use cuda:0 (dev) — embed/layer0 were moved there; model.device may be CPU
            # when calib loads large MoE weights off-GPU (see quant main / 40GB A100 path).
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(ori_dev)
    model.model.embed_tokens = model.model.embed_tokens.to(ori_dev)
    # model.model.norm = model.model.norm.to(ori_dev)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(ori_dev)

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache

    return inps, attention_mask, position_ids, position_embeddings


def enumerate_expert_qconfig(
    qcfg: QLinearConfig, num_layers: int, num_experts: int, expert_id: int, granularity: Literal["expert", "linear"]
)->list[QLayerConfig]:
    '''
    return: [layer_idx, expert_idx, weight_id]
    only the expert_id-th expert in each layer is quantized
    '''
    qlayer_cfg = build_uni_qlayer_cfg(QLinearConfig(), num_experts)
    if granularity == "expert":
        qlayer_cfg.experts[str(expert_id)] = build_uni_qexpert_cfg(qcfg)
        return [qlayer_cfg]
    else:
        res = [deepcopy(qlayer_cfg) for _ in range(3)]
        res[0].experts[str(expert_id)].gate = qcfg
        res[1].experts[str(expert_id)].up = qcfg
        res[2].experts[str(expert_id)].down = qcfg
        return res


class MoeModelQuantizer:
    '''
    quantize the MoE layer
    '''
    def __init__(
        self,
        model:PreTrainedModel, 
        model_id: str,
        qmethod:QMethod,
        calib_data_cache:str=None, # calibration data, e.g. channel smoothing factor for SmoothQuant
        gptq_nsamples=256, # used for GPTQ
        fisher_nsamples=32, # used for Fisher information
        pre_quantized_weight=None, # pre-quantized weight path, used for time-consuming quantization(GPTQ, GPTQ-HAD)
        ori_wcfg:tuple[int,int]=(16,-1), # original weight bitwidth and group size
        online_had:bool=False,
    ):
        self.model_id = model_id
        self.ori_model = model
        self.model_config = model.config
        self.num_layers:int = len(model.model.layers)
        self.model_type: str = self.model_config.model_type
        self.hidden_size: int = self.model_config.hidden_size
        self.dtype = next(iter(model.parameters())).dtype

        assert self.model_type in ["deepseek_v2", "mixtral", "qwen2_moe"], f"Unsupported model type: {self.model_type}"

        self.qmethod: QMethod = qmethod

        if calib_data_cache is not None:
            print(f">>> Loading calibration data from `{calib_data_cache}`...")
            with open(calib_data_cache, "rb") as f:
                self.calib_data_cache = pickle.load(f)
        else:
            self.calib_data_cache = None

        self.fisher_nsamples = fisher_nsamples
        self.ori_wcfg = ori_wcfg

        if self.qmethod in [QMethod.GPTQ_HAD, QMethod.RTN_HAD]:
            # if self.ori_model.model.layers[0].input_layernorm.weight.is_cuda:
            from mxmoe.quant.rotation import ModelRotator
            rotator = ModelRotator(self.ori_model, "hadamard")
            self.online_had_hooks = rotator.rotate_model(self.ori_model, enable_online_rotation=online_had)

        if pre_quantized_weight is None or (isinstance(pre_quantized_weight, str) and self.model_id == "ds2"):
            self.pre_quantized_weight = None
            # Initialize calibration dataset for GPTQ
            if self.qmethod in [QMethod.GPTQ, QMethod.GPTQ_HAD]:
                logger.info(f">>> Enable Online GPTQ weight quantization, may be time-consuming ...")
                from mxmoe.quant.moe_utils import load_tokenizer
                self.gptq_nsamples = gptq_nsamples
                trainloader, _ = get_wikitext2(self.gptq_nsamples, 42, 4096, load_tokenizer(model_id), self.model_id, False)
                self.gptq_layer_state, self.gptq_attention_mask, self.gptq_position_ids, self.gptq_pos_emb = prepare_inps(self.ori_model, trainloader)
                self.gptq_percdamp = 0.01

                logger.info(f"Calibrating GPTQ with {self.gptq_nsamples} samples ...")

        if pre_quantized_weight is not None:
            assert self.qmethod in [QMethod.GPTQ, QMethod.GPTQ_HAD], "Pre-quantized weight now is only used for GPTQ"

            if isinstance(pre_quantized_weight, str):
                if self.qmethod == QMethod.GPTQ_HAD:
                    assert "had" in pre_quantized_weight, "Pre-quantized weight should be hadamard rotated"
                logger.info(f">>> Loading pre-quantized weight from `{pre_quantized_weight}`...")
                self.pre_quantized_weight = torch.load(pre_quantized_weight, map_location="cpu")
                self.pre_quant_tag = "single"
            else:
                assert isinstance(pre_quantized_weight, dict), "Pre-quantized weight should be a dict"
                self.pre_quantized_weight = {}
                self.pre_quant_tag = "mix"
                for qwcfg, qweight_path in pre_quantized_weight.items():
                    logger.info(f">>> Loading pre-quantized weight from `{qweight_path}`...")
                    if self.qmethod == QMethod.GPTQ_HAD:
                        assert "had" in qweight_path, "Pre-quantized weight should be hadamard rotated"
                    self.pre_quantized_weight[qwcfg] = torch.load(qweight_path, map_location="cpu")

        # logger.info(">>> Offloading model to CPU ...")
        # self.model_params: list[dict[str, Tensor]] = move_weight_to_cpu(model)
        # logger.info(">>> Offloading done.")

        self.quantized_model = None
        self.act_quant_hooks: list[torch.utils.hooks.RemovableHandle] = []


    # def __del__(self):
    #     for handle in getattr(self, "online_had_hooks", []): handle.remove()
    #     for handle in self.act_quant_hooks: handle.remove()
    #     if self.ori_model is None and self.quantized_model is None:
    #         return
    #     if self.ori_model is None:
    #         self.ori_model = self.quantized_model
    #     print("Recovering original FULL precision model ...")
    #     recover_weight_from_cpu(self.ori_model, self.model_params)


    @torch.no_grad()
    def quant_model(self, moe_bits_alloc: QModelConfig | QLinearConfig, attn_bits_alloc: QLinearConfig=None):
        '''
        mode_bits_alloc: [layer_idx, expert_idx, weight_id]
            
        attn_bits_alloc:
            now we apply uni quantization for all the weights in attention module (wq, wk, wv, wo)
        '''
        if len(self.act_quant_hooks) > 0:
            for handle in self.act_quant_hooks: handle.remove()

        if self.quantized_model is None:
            self.quantized_model = self.ori_model
            self.ori_model = None
        else:
            raise RuntimeError("Quantized model has been initialized")
            # print(">>> Retrieving original FULL precision model ...")
            # recover_weight_from_cpu(self.quantized_model, self.model_params)
            # torch.cuda.empty_cache()

        logger.info(">>> Quantizing MoE model ...")
        qmodel_experts = get_expert_linears(self.quantized_model, exclude_non_moe_layer=False)

        if isinstance(moe_bits_alloc, QLinearConfig):
            qconfigs = build_uni_qmodel_cfg(moe_bits_alloc, self.num_layers, len(qmodel_experts[1]))
            # if self.model_type == "deepseek_v2": qconfigs.layers[0] = build_uni_qlayer_cfg(moe_bits_alloc, len(qmodel_experts[0]))
            moe_bits_alloc = qconfigs

        self.preprocess_weight(self.quantized_model, -1)

        for layer_i in tqdm(range(self.num_layers), desc="Quantizing MoE layer"):
            if self.model_id == "ds2" and layer_i == 0:
                moe_bits_alloc.layers[str(layer_i)].experts["0"].gate = QLinearConfig(4)
                moe_bits_alloc.layers[str(layer_i)].experts["0"].up = QLinearConfig(4)
                moe_bits_alloc.layers[str(layer_i)].experts["0"].down = QLinearConfig(4)

            self.quant_model_weight_layer(self.quantized_model, layer_i, moe_bits_alloc.layers[str(layer_i)], attn_bits_alloc)
            self.plug_act_quant_hook_layer(self.quantized_model, layer_i, moe_bits_alloc.layers[str(layer_i)], attn_bits_alloc)
        torch.cuda.empty_cache()

        return self.quantized_model


    @torch.no_grad()
    def preprocess_weight(self, model: PreTrainedModel, layer_idx: int, exp_idx: int=-1):
        if self.qmethod in [QMethod.WxA16_NAIVE, QMethod.WxAy_NAIVE, QMethod.GPTQ, QMethod.GPTQ_HAD, QMethod.RTN_HAD]:
            return

        for layer_i in range(self.num_layers):
            if layer_idx != -1 and layer_idx != layer_i: continue

            layer_experts = get_expert_linears(model, layer_idx=layer_i, exclude_non_moe_layer=False)
            for expert_idx, expert in enumerate(layer_experts):
                if exp_idx != -1 and exp_idx != expert_idx: continue

                weight_in_expert = get_linears_in_one_expert(expert)
                for mname, m in weight_in_expert.items():
                    if self.qmethod in [QMethod.SMOOTH_QUANT, QMethod.AWQ]:
                        assert self.calib_data_cache is not None, "Calibration data is required for smoothquant and AWQ"

                        smooth_scale = torch.from_numpy(self.calib_data_cache[layer_i][expert_idx][mname])
                        m.weight.data.mul_(smooth_scale.to(m.weight.device, dtype=m.weight.dtype))
                    else:
                        raise NotImplementedError("")


    @torch.no_grad()
    def quant_model_weight_layer(self, model: PreTrainedModel, layer_idx: int, moe_qconfig: QLayerConfig, attn_qconfig: QLinearConfig=None):
        '''
        moe_qconfig: [layer_idx, expert_idx]
            
        attn_qconfig:
            now we quantize all the attention weights (wq, wk, wv, wo) to the same bitwidth
        '''
        ds2_flag = (self.model_id == "ds2" and layer_idx == 0 and isinstance(self.pre_quantized_weight, str))
        # print(f"Quantizing layer-{layer_idx} ...")
        if self.pre_quantized_weight is not None and not ds2_flag:
            assert self.qmethod in [QMethod.GPTQ, QMethod.GPTQ_HAD], "Pre-quantized weight is only used for GPTQ"

            if self.pre_quant_tag == "single":
                for expert_idx, qexpert_cfg in moe_qconfig.experts.items():
                    for linear_block, qlinear_cfg in qexpert_cfg.qmap().items():
                        if qlinear_cfg.w_bits >= 16: continue
                        substitue_moe_weights(self.model_id, model, self.pre_quantized_weight, layer_idx, int(expert_idx), linear_block)
            elif self.pre_quant_tag == "mix":
                layer_experts = get_expert_linears(model, layer_idx, exclude_non_moe_layer=False)
                num_experts = len(layer_experts)

                for expert_idx, qexpert_cfg in moe_qconfig.experts.items():
                    if int(expert_idx) >= num_experts: break
                    for linear_block, qlinear_cfg in qexpert_cfg.qmap().items():
                        cur_qwcfg = (qlinear_cfg.w_bits, qlinear_cfg.w_gsize, qlinear_cfg.w_sym)
                        if cur_qwcfg == ori_wcfg : continue
                        qweight = self.pre_quantized_weight[cur_qwcfg]
                        new_weight = get_linear_block_weight(model_id, qweight, layer_idx, expert_idx, linear_block)
                        old_weight = get_linear_block_weight(model_id, model, layer_idx, expert_idx, linear_block)
                        old_weight.copy_(new_weight)
            else:
                raise NotImplementedError("")

            # TODO: 2. quantize the weights of attention layer
            return

        for layer_i in range(self.num_layers):
            if layer_idx != -1 and layer_idx != layer_i: continue
            cur_layer = model.model.layers[layer_i]

            layer_experts = get_expert_linears(model, layer_idx=layer_i, exclude_non_moe_layer=False)
            # 1. quantize the weights of moe layer
            for expert_idx, expert in enumerate(layer_experts):
                qmap = moe_qconfig.experts[str(expert_idx)].qmap()
                weights_in_expert = get_linears_in_one_expert(expert)

                for linear_block, m in weights_in_expert.items():
                    cfg = qmap[linear_block]
                    if cfg.w_bits >= 16: continue
                    # GPTQ
                    if self.qmethod in [QMethod.GPTQ, QMethod.GPTQ_HAD]:
                        print(f"Quantizing layer-{layer_i} expert-{expert_idx} block-{linear_block} with GPTQ ...")
                        gptq = GPTQ(m)
                        gptq.quantizer = GPTQuantizer()
                        gptq.quantizer.configure(
                            cfg.w_bits, perchannel=True, sym=cfg.w_sym, mse=False
                        )
                        handle: torch.utils.hooks.RemovableHandle = []
                        handle = m.register_forward_hook(lambda _, inp, out: gptq.add_batch(inp[0].data, out.data))
                        # if self.gptq_attention_mask
                        for j in range(self.gptq_nsamples):
                            self.gptq_layer_state[j] = cur_layer(
                                self.gptq_layer_state[j].unsqueeze(0),
                                attention_mask=self.gptq_attention_mask,
                                position_ids=self.gptq_position_ids,
                                position_embeddings=self.gptq_pos_emb,
                            )[0]
                        handle.remove()
                        gptq.fasterquant(
                            percdamp=self.gptq_percdamp, groupsize=cfg.w_gsize, actorder=True, static_groups=True
                        )
                        gptq.free()
                        continue
                    # RTN
                    quantizer = Quantizer(cfg.w_bits, cfg.w_sym, cfg.w_gsize, cfg.w_clip)
                    m.weight.data = quantizer.fake_quant(m.weight.data)
                    if self.qmethod == QMethod.AWQ:
                        awq_scale = torch.from_numpy(self.calib_data_cache[layer_i][expert_idx][linear_block]).to(dtype=m.dtype, device=m.device)
                        m.weight.data.div_(awq_scale)

            # TODO: 2. quantize the weights of attention layer
                

    def plug_act_quant_hook_layer(self, model: PreTrainedModel, layer_idx: int, moe_qconfig: QLayerConfig, attn_qconfig: QLinearConfig=None):
        layer_act_quant_hooks = []
        for layer_i in range(self.num_layers):
            if layer_idx != -1 and layer_idx != layer_i: continue
            
            layer_experts = get_expert_linears(model, layer_idx=layer_i, exclude_non_moe_layer=False)

            # 1. quantize the act of moe layer
            for expert_idx, expert in enumerate(layer_experts):
                qmap = moe_qconfig.experts[str(expert_idx)].qmap()
                weights_in_expert = get_linears_in_one_expert(expert)

                for mname, m in weights_in_expert.items():
                    cfg: QLinearConfig = qmap[mname]
                    if cfg.a_bits >= 16: continue
                    quantizer = Quantizer(cfg.a_bits, cfg.a_sym, cfg.a_gsize, cfg.a_clip)

                    if self.qmethod == QMethod.SMOOTH_QUANT:
                        assert self.calib_data_cache is not None, "Calibration data is required for smoothquant"

                        smooth_scale = torch.from_numpy(self.calib_data_cache[layer_i][expert_idx][mname])
                        act_quant_hook = partial(
                            smooth_act_quant_hook,
                            quantizer=quantizer,
                            smooth_scale=smooth_scale.to(m.weight.dtype).to(m.weight.device)
                        )
                    elif self.qmethod in [QMethod.WxAy_NAIVE, QMethod.GPTQ, QMethod.GPTQ_HAD, QMethod.RTN_HAD]:
                        act_quant_hook = partial(mlp_inp_quant_hook, quantizer=quantizer)
                    elif self.qmethod == QMethod.LLM_INT8:
                        raise NotImplementedError("")
                    else:
                        raise NotImplementedError("")

                    # print(f"Plugging act quant hook for layer-{layer_i} expert-{expert_idx} block-{mname} ...")
                    handle = m.register_forward_pre_hook(act_quant_hook)
                    layer_act_quant_hooks.append(handle)


            # TODO: 2. quantize the input act of attention layer
            pass
        
            # TODO: 3. low-precision attention
            pass

        self.act_quant_hooks.extend(layer_act_quant_hooks)


    def get_model_quant_error(
        self,
        dataloader: list[Tensor],
        metric: Literal["layer_out_norm", "model_out_norm", "fisher"],
        granularity: Literal["expert", "linear"],
        save_path: str,
        moe_bits_alloc: QLinearConfig,
        attn_bits_alloc: QLinearConfig=None,
    ) -> list[list[float]] | list[float]:
        # Prefer a real parameter device: multi-device HF maps may not expose .device cleanly.
        try:
            ori_dev = self.ori_model.device
        except Exception:
            ori_dev = next(self.ori_model.parameters()).device
        # layer_out_norm streams one layer to cuda:0; model may live on CPU to fit 40GB GPUs.
        if self.model_id in ["qwen2_moe_57b", "mixtral"]:
            assert ori_dev.type == "cpu"
        else:
            assert ori_dev.type in ("cuda", "cpu"), f"unexpected device {ori_dev}"

        assert self.quantized_model is None, "Quantized model should not be initialized"

        # do not use cache
        model_use_cache = self.ori_model.config.use_cache
        self.ori_model.config.use_cache = False

        num_samples = len(dataloader)
        input_len = dataloader[0].shape[1]

        # allocate for quant error
        layer_loss: list[list[list[float]]] = [[] for _ in range(self.num_layers)]
        layer_loss_save = {}
        
        layers: list[nn.Module] = self.ori_model.model.layers
        if metric == "layer_out_norm":
            dev = torch.device("cuda:0")

            inps, attn_mask, position_ids, pos_emb = prepare_inps(self.ori_model, dataloader)
            # Keep only one float64 buffer on GPU. Per-sample Frobenius accumulation is
            # mathematically equal to torch.norm(q-fp) and avoids a second 8GiB tensor + temp.
            full_precision_outs: Tensor = torch.zeros_like(inps).to(torch.float64)

            for layer_idx in tqdm(range(len(layers))):
                self.ori_model.model.layers[layer_idx] = layers[layer_idx].to(dev)
                cpu_weights_copy = offload_moe_weights(self.model_id, self.ori_model, layer_idx)

                layer = self.ori_model.model.layers[layer_idx]

                with torch.inference_mode():
                    # 1. get the output of the full precision layer
                    for i in range(num_samples):
                        full_precision_outs[i] = layer(
                            inps[i].unsqueeze(0),
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            position_embeddings=pos_emb,
                        )[0].to(torch.float64)

                # 2. get the output of the quantized layer
                num_layer_experts = len(get_expert_linears(self.ori_model, layer_idx, exclude_non_moe_layer=False))
                for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                    qlayer_cfgs = enumerate_expert_qconfig(moe_bits_alloc, self.num_layers, num_layer_experts, exp_id, granularity)

                    expert_err = []
                    for qlinear_block, qlayer_cfg in zip(["gate", "up", "down"], qlayer_cfgs):
                        with torch.inference_mode():
                            # self.preprocess_weight(self.ori_model, layer_idx, exp_idx=exp_id)
                            if self.pre_quantized_weight is not None:
                                substitue_moe_weights(self.model_id, self.ori_model, self.pre_quantized_weight, layer_idx, exp_id, qlinear_block)
                            else:
                                self.quant_model_weight_layer(self.ori_model, layer_idx, qlayer_cfg, attn_bits_alloc)
                            self.plug_act_quant_hook_layer(self.ori_model, layer_idx, qlayer_cfg, attn_bits_alloc)
                            # 3. quantization error (chunked Frobenius norm; same as full-tensor norm)
                            err_sq = 0.0
                            for i in range(num_samples):
                                qout = layer(
                                    inps[i].unsqueeze(0),
                                    attention_mask=attn_mask,
                                    position_ids=position_ids,
                                    position_embeddings=pos_emb,
                                )[0].to(torch.float64)
                                err_sq += torch.norm(qout - full_precision_outs[i]).item() ** 2
                            quant_err = err_sq ** 0.5
                            expert_err.append(quant_err)

                            # 4. recover the FULL precision layer
                            substitue_moe_weights(self.model_id, self.ori_model, cpu_weights_copy, layer_idx, exp_id, qlinear_block)
                            for handle in self.act_quant_hooks: handle.remove()

                    layer_loss[layer_idx].append(expert_err)
                    logger.info(f"{self.model_id} L{layer_idx}-E{exp_id} {moe_bits_alloc} (layer_out_norm): {expert_err}")

                del cpu_weights_copy
                self.ori_model.model.layers[layer_idx] = layers[layer_idx].to(ori_dev)
                torch.cuda.empty_cache()

                # 5. serialization
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    layer_loss_save[layer_idx] = {e: layer_loss[layer_idx][e] for e in range(len(layer_loss[layer_idx]))}
                    json.dump(layer_loss_save, f)
                logger.info(f"Layer-{layer_idx} quant error(layer_out_norm):\n{layer_loss[layer_idx]}")


                # 6. prepare the inputs for next layer
                inps, full_precision_outs = full_precision_outs.to(dtype=inps.dtype, device=dev), full_precision_outs

        elif metric == "model_out_norm":
            # inps: [num_samples, seqlen]
            inps = torch.vstack([d.to(ori_dev) for d in dataloader])

            # outs: [num_samples, seqlen, hidden_size]
            full_precision_outs: Tensor = torch.zeros((num_samples, inps.shape[-1], self.hidden_size), device=next((layers[-1]).parameters()).device, dtype=torch.float64)
            quantized_outs: Tensor = torch.zeros_like(full_precision_outs)
            # 1. get the output of full-precision model (hidden_states of the last layer)
            with torch.inference_mode():
                for i in tqdm(range(num_samples), desc="Getting full precision output"):
                    full_precision_outs[i] = self.ori_model.model(inps[i:i+1,:]).last_hidden_state.to(torch.float64)

            # 2. get the model output of with quantized layer
            for layer_idx in tqdm(range(len(layers)), desc="Getting quantized output"):
                num_layer_experts = len(get_expert_linears(self.ori_model, layer_idx, exclude_non_moe_layer=False))
                for exp_id in tqdm(range(num_layer_experts), leave=False, desc="Quantizing Expert"):
                    qlayer_cfgs = enumerate_expert_qconfig(moe_bits_alloc, self.num_layers, num_layer_experts, exp_id, granularity)

                    expert_err = []
                    for qlinear_block, qlayer_cfg in zip(["gate", "up", "down"], qlayer_cfgs):
                        ori_weight = get_linear_block_weight(self.model_id, self.ori_model, layer_idx, exp_id, qlinear_block).cpu()
                        self.preprocess_weight(self.ori_model, layer_idx, exp_idx=exp_id)
                        self.quant_model_weight_layer(self.ori_model, layer_idx, qlayer_cfg, attn_bits_alloc)
                        self.plug_act_quant_hook_layer(self.ori_model, layer_idx, qlayer_cfg, attn_bits_alloc)

                        with torch.inference_mode():
                            for i in range(num_samples):
                                quantized_outs[i] = self.ori_model.model(inps[i:i+1,:]).last_hidden_state.to(torch.float64)

                            # 3. calculate the quantization error
                            # quant_err = F.mse_loss(quantized_outs, full_precision_outs).item()
                            quant_err = torch.norm(quantized_outs.sub_(full_precision_outs)).item()
                            expert_err.append(quant_err)

                            # 4. recover the FULL precision layer
                            for handle in self.act_quant_hooks: handle.remove()
                            qweight = get_linear_block_weight(self.model_id, self.ori_model, layer_idx, exp_id, qlinear_block)
                            qweight.copy_(ori_weight)

                    layer_loss[layer_idx].append(expert_err)
                    print(f"Layer-{layer_idx}-{exp_id} quant error(model_out_norm): {expert_err}")

                # 5. serialization
                with open(save_path, "w") as f:
                    layer_loss_save[layer_idx] = {e: layer_loss[layer_idx][e] for e in range(len(layer_loss[layer_idx]))}
                    json.dump(layer_loss_save, f)
                print(f"Layer-{layer_idx} quant error(layer_out_norm): {layer_loss[layer_idx]}")

        elif metric == "fisher":
            raise NotImplementedError("")

        else:
            raise ValueError(f"Unknown metric: {metric}")


        self.ori_model.config.use_cache = model_use_cache

        return layer_loss


    def get_ori_model(self):
        return self.ori_model


    def get_quant_model(self):
        return self.quantized_model


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    """
    e.g.
    1. CALIBRATION
        get the quant loss of each linear-blocks of qwen1.5_moe under RTN w4a4_g-1_sym quantization config:
        CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.quant calib --model qwen2_moe --method rtn --metric layer_out_norm --qcfg w4a4_g-1_sym

        get the quant loss of each linear-blocks of deepseekV2-Lite under GPTQ-Hadamard w4a4_g-1_sym quantization config:
        CUDA_VISIBLE_DEVICES=1 python -m mxmoe.quant.quant calib --model ds2 --method gptq-had --metric layer_out_norm --qcfg w4a4_g-1_sym

    2. EVALUATION
        evaluate the performance of qwen1.5_moe under RTN w4a4_g-1_sym quantization config:
        CUDA_VISIBLE_DEVICES=2 python -m mxmoe.quant.quant eval --model qwen2_moe --method rtn-had --qstr w4a4_g-1_sym --tasks ppl
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", dest="sub")

    parser_eval = subparsers.add_parser("eval", help="Eval quantized model")
    parser_eval.add_argument("--model", type=str, default="qwen2_moe", help="Model ID")
    parser_eval.add_argument("--method", type=str, default="rtn", choices=["rtn", "smooth", "gptq", "gptq-had", "rtn-had"], help="Quantization method")
    parser_eval.add_argument("--qweight", type=str, help="Load the quantized weight from file")
    parser_eval.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser_eval.add_argument("--qconfig", type=str, help="Quantization config file")
    parser_eval.add_argument("--tasks", type=str, nargs="+", default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada"], help="Evaluation tasks")
    parser_eval.add_argument("--qstr", type=str, help="Quantization config str")
    parser_eval.add_argument("--online_had", action="store_true", help="Whether to use use Hadamard transform")
    parser_eval.add_argument("--bs", type=int, default=128, help="zero-shot task eval batch")
    parser_eval.add_argument("--save", type=str, help="Save path")
    # Single-GPU fit: accelerate device_map="auto" + max_memory (layer placement / CPU spill).
    # Example 3090 (24GB): --max-gpu-memory 16GiB --max-cpu-memory 64GiB
    # Keep ~8GiB free for activations: 4k-token PPL logits on a 151k vocab need ~5GiB+.
    # Do not .to() the whole model after load. Not for calib layer_out_norm / online GPTQ.
    parser_eval.add_argument(
        "--max-gpu-memory", type=str, default=None,
        help="GPU memory budget for accelerate placement (e.g. 16GiB on a 24GB 3090). "
        "Enables CPU spill of remaining layers. Leave ~8GiB free for activations/logits.",
    )
    parser_eval.add_argument(
        "--max-cpu-memory", type=str, default=None,
        help="CPU memory budget for accelerate placement (e.g. 64GiB). Defaults to 128GiB if only --max-gpu-memory is set.",
    )
    parser_eval.add_argument(
        "--offload-folder", type=str, default=None,
        help="Directory for accelerate disk offload when modules do not fit in GPU+CPU budgets.",
    )

    parser_calib = subparsers.add_parser("calib", help="Calibrate for quantized model")
    parser_calib.add_argument("--model", type=str, default="ds2", help="Model ID")
    parser_calib.add_argument("--method", type=str, default="rtn", choices=["rtn", "smooth", "gptq", "gptq-had", "rtn-had"], help="Quantization method")
    parser_calib.add_argument("--qcfg", type=str, required=True, help="Quantization config str")
    parser_calib.add_argument("--metric", type=str, choices=["fisher", "model_out_norm", "layer_out_norm"], help="Evaluation metric")
    parser_calib.add_argument("--qweight", type=str, default=None, help="Load the quantized weight from file")
    parser_calib.add_argument("--gran", type=str, choices=["expert", "linear"], default="linear", help="Granularity of estimation")
    parser_calib.add_argument("--online_had", action="store_true", help="Whether to use use Hadamard transform")
    parser_calib.add_argument("--nsamples", type=int, default=128, help="Number of samples for calibration")
    parser_calib.add_argument("--seed", type=int, default=42, help="Random Seed.")

    args = parser.parse_args(
        # [
        #     "calib",
        #     "--model", "mixtral",
        #     "--method", "gptq-had",
        #     "--qcfg", "w1a16_g128_asym",
        #     "--qweight", "./out/gptq/mixtral-w1_g128_had_n256.pt",
        #     "--metric", "layer_out_norm",
        # ]

        # [
        #     "eval",
        #     "--model", "mixtral",
        #     "--method", "gptq-had",
        #     "--tasks", "ppl",
        #     "--qstr", "w2a16_g128_asym",
        #     # "--qweight", "./out/gptq/qwen2_moe_57b-w4_g-1_had_n256.pt",
        # ]
    )

    seed_everything(args.seed)

    model_id = args.model
    quant_type = args.method

    ############################################################################
    qtype_2_method = {
        "rtn": QMethod.WxAy_NAIVE,
        # "smooth": QMethod.SMOOTH_QUANT,
        "gptq": QMethod.GPTQ,
        "gptq-had": QMethod.GPTQ_HAD,
        "rtn-had": QMethod.RTN_HAD,
    }
    qmethod = qtype_2_method[quant_type]

    model_name = ID2NAME[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quant_type in ["gptq-had", "rtn-had"]:
        qweight_map = {
            (8,-1, True):  f"{CUR_DIR}/out/gptq/{model_id}-w8_g-1_sym_had_n256.pt",
            (4,-1, True):  f"{CUR_DIR}/out/gptq/{model_id}-w4_g-1_sym_had_n256.pt",
            (4,128, True): f"{CUR_DIR}/out/gptq/{model_id}-w4_g128_sym_had_n256.pt",
            (8,-1, False): f"{CUR_DIR}/out/gptq/{model_id}-w8_g-1_had_n256.pt",
            (4,-1, False): f"{CUR_DIR}/out/gptq/{model_id}-w4_g-1_had_n256.pt",
            (4,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w4_g128_had_n256.pt",
            (3,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w3_g128_had_n256.pt",
            (2,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w2_g128_had_n256.pt",
            (1,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w1_g128_had_n256.pt",
        }
    else:
        qweight_map = {
            (8,-1, True):  f"{CUR_DIR}/out/gptq/{model_id}-w8_g-1_sym_n256.pt",
            (4,-1, True):  f"{CUR_DIR}/out/gptq/{model_id}-w4_g-1_sym_n256.pt",
            (4,128, True): f"{CUR_DIR}/out/gptq/{model_id}-w4_g128_sym_n256.pt",
            (8,-1, False): f"{CUR_DIR}/out/gptq/{model_id}-w8_g-1_n256.pt",
            (4,-1, False): f"{CUR_DIR}/out/gptq/{model_id}-w4_g-1_n256.pt",
            (4,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w4_g128_n256.pt",
            (3,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w3_g128_n256.pt",
            (2,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w2_g128_n256.pt",
            (1,128,False): f"{CUR_DIR}/out/gptq/{model_id}-w1_g128_n256.pt",
        }

    ori_wcfg = (16,-1)
    # Calib layer_out_norm streams one decoder layer to cuda:0; keep weights on CPU so
    # qwen2_moe (~28GiB bf16) + float64 activation buffers fit on 40GB A100s.
    # (Hand layerwise calib path; not the accelerate fit-map used by eval placement.)
    if model_id in ["qwen2_moe", "qwen2_moe_57b", "mixtral"] and args.sub == "calib":
        print(f">>> load model to CPU (calib sequential layer offload)")
        model_name = ID2NAME[model_id]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )
    else:
        # Eval (and small-model calib): optional accelerate GPU+CPU layer placement.
        max_memory = None
        offload_folder = None
        if args.sub == "eval":
            max_memory = build_max_memory(args.max_gpu_memory, args.max_cpu_memory)
            offload_folder = args.offload_folder
            if (max_memory is not None or offload_folder is not None) and (
                args.qstr is not None or args.qconfig is not None or args.qweight is not None
            ):
                # accelerate hooks keep offloaded (CPU/disk) weights in a side map and
                # restore them every forward, so in-place quantization / substitution on
                # those layers is silently dropped -> mixed-precision garbage results.
                raise SystemExit(
                    "--max-gpu-memory/--max-cpu-memory/--offload-folder (fit-map) cannot be "
                    "combined with --qstr/--qconfig/--qweight: in-place quantization is "
                    "silently discarded on offloaded layers. Fit-map is for full-precision "
                    "whole-model eval only; quantized checkpoints are smaller — load them "
                    "without a memory budget."
                )
            if max_memory is not None or offload_folder is not None:
                print(
                    ">>> load model with accelerate placement "
                    f"(max_memory={max_memory}, offload_folder={offload_folder})"
                )
        model = load_hf_model(
            model_id,
            max_memory=max_memory,
            offload_folder=offload_folder,
        )
    ############################################################################

    if args.sub == "eval":
        ts = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        logger = setup_logger("quant-eval", log_file=f"{CUR_DIR}/log/evaluation_{ts}.log")

        if args.qconfig is not None:
            qmodel_config = build_qmodel_cfg_from_json(args.qconfig)
            all_bits = get_all_wbits(qmodel_config)
            if ori_wcfg in all_bits: all_bits.remove(ori_wcfg)

            qweight_map = {k: v for k, v in qweight_map.items() if k in all_bits}

            logger.info(f">>> Build Qmodel from Qconfig: {args.qconfig}")
            logger.info(f">>> Pre-Quantized weight: {all_bits}")

            pre_quantized_weight = None if args.qweight is None else qweight_map
            model_quantizer = MoeModelQuantizer(model, model_id, qmethod, pre_quantized_weight=pre_quantized_weight, ori_wcfg=ori_wcfg, online_had=args.online_had)
            qmodel = model_quantizer.quant_model(qmodel_config)
        elif args.qstr is not None:
            assert args.qconfig is None
            qstr = args.qstr
            wbits = int(qstr.split("w")[1].split("a")[0])
            abits = int(qstr.split("a")[1].split("_")[0])
            gsize = int(qstr.split("g")[1].split("_")[0])
            sym   = bool("asym" not in qstr)
            qmodel_config = build_uni_qmodel_cfg(
                QLinearConfig(w_bits=wbits, w_sym=sym, w_gsize=gsize, a_bits=abits, a_sym=sym, a_gsize=gsize),
                model.config.num_hidden_layers,
                68
            )

            logger.info(f">>> Build Qmodel from uni-Qconfig: {qstr}")

            pre_quantized_weight = None if args.qweight is None else args.qweight
            model_quantizer = MoeModelQuantizer(model, model_id, qmethod, ori_wcfg=ori_wcfg, pre_quantized_weight=pre_quantized_weight, online_had=args.online_had)
            qmodel = model_quantizer.quant_model(qmodel_config)
        else:
            qmodel = model

        evaluator = Evaluator(tokenizer, model_id)
        eval_res = {}

        if "ppl" in args.tasks:
            eval_res["ppl"] = evaluator.eval_ppl(qmodel)
            args.tasks.remove("ppl")

        if len(args.tasks) != 0:
            logger.info(f"ZeroShot Tasks: {args.tasks}")
            eval_res["zero_shot"] = evaluator.eval_tasks(qmodel, task_list=args.tasks, bs=args.bs)
        
        logger.info(f"Eval Results: {eval_res}")

        if args.save:
            logger.info(f"Save to `{args.save}`")
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            with open(args.save, "w") as f:
                json.dump(eval_res, f)

        # evaluator = Evaluator(tokenizer, model_id)
        # model_quantizer = MoeModelQuantizer(model, model_id, qmethod)
        # uni_wquant = {}
        # for w_gsize in [128, -1]:
        # # for w_gsize in [-1]:
        #     uni_wquant[w_gsize] = {}
        #     for wbits in [8, 4, 3, 2]:
        #     # for wbits in [16]:
        #         uni_qconfig = QLinearConfig(w_bits=wbits, w_gsize=w_gsize, w_sym=False, w_clip=(1.0, 1.0))
        #         print(uni_qconfig)
        #         uni_wquant[w_gsize][wbits] = evaluator.eval_ppl(model_quantizer.quant_model(uni_qconfig))
        # print(f"uni_w-quant: {uni_wquant}")
        # del model_quantizer


        # model_quantizer = MoeModelQuantizer(model, model_id, qmethod)
        # uni_wactquant = {}
        # for w_act_bits in [(8, 8), (6, 8), (6, 6), (4, 8), (4, 4)]:
        #     uni_qconfig = QLinearConfig(w_bits=w_act_bits[0],w_sym=True, a_bits=w_act_bits[1], a_sym=True)
        #     print(uni_qconfig)
        #     uni_wactquant[w_act_bits] = evaluator.eval_ppl(model_quantizer.quant_model(uni_qconfig))
        # print(f"uni_w-act-quant: {uni_wactquant}")

        # del model_quantizer
        exit(0)

    ############################################################################
    if args.sub == "calib":
        nsamples = args.nsamples
        seqlen = 4096

        ts = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        logger = setup_logger("quant-calib", log_file=f"{CUR_DIR}/log/calib_qerror_{ts}.log")

        trainloader, testloader = get_wikitext2(nsamples, 42, seqlen, tokenizer, model_id=model_id, test_only=False)

        uni_attn_weight_cfg = QLinearConfig(w_bits=16)

        str_to_qcfg = {
            "w8a8_g-1_sym": QLinearConfig(w_bits=8,w_sym=True,w_gsize=-1, a_bits=8,a_sym=True,a_gsize=-1),
            "w4a4_g-1_sym": QLinearConfig(w_bits=4,w_sym=True,w_gsize=-1, a_bits=4,a_sym=True,a_gsize=-1),
            "w4a4_g128_sym": QLinearConfig(w_bits=4,w_sym=True,w_gsize=128, a_bits=4,a_sym=True,a_gsize=128),
            "w8a16_g-1_asym": QLinearConfig(w_bits=8,w_sym=False,w_gsize=-1),
            "w4a16_g-1_asym": QLinearConfig(w_bits=4,w_sym=False,w_gsize=-1),
            "w4a16_g128_asym": QLinearConfig(w_bits=4,w_sym=False,w_gsize=128),
            "w3a16_g128_asym": QLinearConfig(w_bits=3,w_sym=False,w_gsize=128),
            "w2a16_g128_asym": QLinearConfig(w_bits=2,w_sym=False,w_gsize=128),
            "w1a16_g128_asym": QLinearConfig(w_bits=1,w_sym=False,w_gsize=128),
        }
        uni_qconfig = str_to_qcfg[args.qcfg]

        metric = args.metric
        logger.info(f">>> Calibration Quant Error with `{qmethod}`")
        logger.info(f">>> MoE Config[{uni_qconfig}]")
        logger.info(f">>> Attn Config[{uni_attn_weight_cfg}]")
        logger.info(f">>> Metric: `{metric}`, Granularity: `{args.gran}`")

        if quant_type == "rtn":
            model_quantizer = MoeModelQuantizer(model, model_id, QMethod.WxAy_NAIVE)
            save_path = f"{CUR_DIR}/calib/{model_id}-MOE-rtn-{uni_qconfig}-{'wiki2'}-{nsamples}-{seqlen}-{metric}.json"
        elif quant_type == "gptq":
            assert uni_qconfig.a_bits == 16
            pre_quantized_weight = args.qweight
            model_quantizer = MoeModelQuantizer(model, model_id, QMethod.GPTQ, pre_quantized_weight=pre_quantized_weight)
            save_path = f"{CUR_DIR}/calib/{model_id}-MOE-gptq-{uni_qconfig}-{'wiki2'}-{nsamples}-{seqlen}-{metric}.json"
        elif quant_type == "gptq-had":
            pre_quantized_weight = args.qweight
            model_quantizer = MoeModelQuantizer(model, model_id, QMethod.GPTQ_HAD, pre_quantized_weight=pre_quantized_weight, online_had=args.online_had)
            save_path = f"{CUR_DIR}/calib/{model_id}-MOE-gptq-had-{uni_qconfig}-{'wiki2'}-{nsamples}-{seqlen}-{metric}.json"
        elif quant_type == "smooth":
            raise NotImplementedError("")
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")

        model_quant_loss = model_quantizer.get_model_quant_error(
            trainloader, metric,
            args.gran, save_path,
            uni_qconfig, uni_attn_weight_cfg,
        )
        # print(model_quant_loss)
