# SPDX-License-Identifier: Apache-2.0
"""Native MxMoE experts: exact CT W2/W4/W8 g128 weights, FP16 interface.

Optional qconfig-driven A4/A8 preparation lives inside this operator. Gate and
Up share input preparation only when their complete activation schemes match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    get_moe_permute_scratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

from mxmoe.vllm import ops
from mxmoe.vllm.activation import DeviceActivationPlan, load_activation_plan

GROUP_SIZE = 128
WEIGHT_DTYPE_BY_BITS = {2: "int2", 4: "int4", 8: "int8"}


def _single_gpu(parallel: FusedMoEParallelConfig) -> bool:
    return (
        parallel.tp_size == parallel.dp_size == parallel.pcp_size
        == parallel.sp_size == parallel.ep_size == 1
        and not parallel.use_ep and not parallel.enable_eplb
    )


def mxmoe_unsupported_reason(
    moe_config: FusedMoEConfig, quant_config, *, may_have_zp: bool, may_have_bias: bool,
) -> str | None:
    """Validate deployment once; use real checkpoint group size, not a key sentinel."""
    if not MxMoEExperts._supports_current_device():
        return "mxmoe requires CUDA SM80 or newer"
    if moe_config.in_dtype != torch.float16:
        return "mxmoe requires FP16 activations; select dtype=half, not BF16"
    if not _single_gpu(moe_config.moe_parallel_config):
        return "mxmoe requires one GPU, TP1, and EP/EPLB off"
    if moe_config.is_lora_enabled or may_have_bias or moe_config.has_bias:
        return "mxmoe does not support LoRA or expert bias"
    if moe_config.activation != MoEActivation.SILU or not moe_config.is_act_and_mul:
        return "mxmoe requires gated SiLU"
    if any(value is not None for value in (
        moe_config.swiglu_limit, moe_config.swiglu_alpha, moe_config.swiglu_beta,
        moe_config.activation_situ_beta, moe_config.activation_situ_linear_beta,
    )):
        return "mxmoe does not support scaled/clamped or SituGLU activations"
    hidden, intermediate = moe_config.hidden_dim, moe_config.intermediate_size_per_partition
    if hidden % GROUP_SIZE or intermediate % GROUP_SIZE:
        return "mxmoe requires hidden and expert intermediate dimensions divisible by 128"
    if (moe_config.hidden_dim_unpadded not in (None, hidden)
            or moe_config.intermediate_size_per_partition_unpadded not in (None, intermediate)
            or moe_config.intermediate_pad not in (None, 0)):
        return "mxmoe does not support runtime expert padding"
    if not isinstance(quant_config, QuantizationArgs):
        return "mxmoe requires compressed-tensors weight quantization metadata"
    if quant_config.num_bits not in WEIGHT_DTYPE_BY_BITS:
        return "mxmoe supports W2/W4/W8 only"
    if quant_config.symmetric or not may_have_zp:
        return "mxmoe requires asymmetric weights with raw zero points"
    if quant_config.group_size != GROUP_SIZE or quant_config.strategy != QuantizationStrategy.GROUP:
        return "mxmoe requires group128 weights"
    if quant_config.dynamic or quant_config.actorder is not None:
        return "mxmoe requires static weights without activation ordering"
    return None


@dataclass
class MxMoEQuantConfig(FusedMoEQuantConfig):
    # The public activation descriptors remain FP16: quantization is deferred
    # to the experts and cannot be represented by one uniform a1/a2 descriptor.
    activation_plan: DeviceActivationPlan | None = None


def make_mxmoe_moe_quant_config(
    w1_scale: torch.Tensor, w2_scale: torch.Tensor, num_bits: int, group_size: int,
    activation_plan: DeviceActivationPlan | None = None,
) -> FusedMoEQuantConfig:
    if num_bits not in WEIGHT_DTYPE_BY_BITS or group_size != GROUP_SIZE:
        raise ValueError("mxmoe requires W2/W4/W8 group128")
    # Raw zeros are the second member of scale_zero; W2 really is int2.
    return MxMoEQuantConfig(
        _a1=FusedMoEQuantDesc(), _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(dtype=WEIGHT_DTYPE_BY_BITS[num_bits],
                            shape=GroupShape(1, group_size), scale=w1_scale),
        _w2=FusedMoEQuantDesc(dtype=WEIGHT_DTYPE_BY_BITS[num_bits],
                            shape=GroupShape(1, group_size), scale=w2_scale),
        activation_plan=activation_plan,
    )


def convert_mxmoe_weights(
    layer: torch.nn.Module, quant_config, w13: torch.Tensor, w2: torch.Tensor,
    w13_scale: torch.Tensor, w2_scale: torch.Tensor,
    w13_qzeros: torch.Tensor | None, w2_qzeros: torch.Tensor | None,
    w13_bias: torch.Tensor | None, w2_bias: torch.Tensor | None,
):
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("mxmoe does not support router-weight-on-input")
    if w13_bias is not None or w2_bias is not None:
        raise ValueError("mxmoe does not support expert bias")
    if w13_qzeros is None or w2_qzeros is None:
        raise ValueError("mxmoe requires raw packed zero points")
    if w13.shape[1] % (2 * GROUP_SIZE) or w2.shape[1] % GROUP_SIZE:
        raise ValueError("mxmoe requires separately aligned gate/up and down projections")
    bits = quant_config.num_bits
    from vllm.config import get_current_vllm_config

    path = get_current_vllm_config().kernel_config.mxmoe_activation_config
    device_plan = None
    if path is not None:
        plan = load_activation_plan(
            path, layer.layer_name, w13.shape[0], bits,
            quant_config.group_size, quant_config.symmetric,
        )
        if plan.quantized:
            device_plan = plan.to_device(w13.device)
    layer.mxmoe_activation_plan = device_plan
    # Each projection occupies the same packed storage span in both layouts.
    # A16 keeps the old N-packed layout; A4/A8 uses K-packed centered codes.
    if device_plan is None:
        packed13, scales13 = ops.prepare_weights(w13, w13_scale, w13_qzeros, bits)
        packed2, scales2 = ops.prepare_weights(w2, w2_scale, w2_qzeros, bits)
    else:
        packed13, scales13 = ops.prepare_weights_mixed(
            w13, w13_scale, w13_qzeros, bits, device_plan.bits[:2],
        )
        packed2, scales2 = ops.prepare_weights_mixed(
            w2, w2_scale, w2_qzeros, bits, device_plan.bits[2:3],
        )
    return packed13, packed2, scales13, scales2, None, None, None, None, None, None


class MxMoEExperts(mk.FusedMoEExpertsModular):
    """Standard-format experts; GEMM scratch is owned per execution slot."""

    requires_interleaved_w13 = False

    def __init__(self, moe_config, quant_config, max_num_tokens=None, num_dispatchers=None):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        bits_by_dtype = {dtype: bits for bits, dtype in WEIGHT_DTYPE_BY_BITS.items()}
        self.num_bits = bits_by_dtype[quant_config.weight_quant_dtype]
        self.group_size = GROUP_SIZE
        if (quant_config.quant_dtype is not None
                or quant_config._w1.shape != GroupShape(1, GROUP_SIZE)
                or quant_config._w2.shape != GroupShape(1, GROUP_SIZE)):
            raise ValueError("mxmoe requires an FP16 interface and group128 scale_zero tensors")
        self.activation_plan = quant_config.activation_plan
        experts, hidden = moe_config.num_local_experts, moe_config.hidden_dim
        intermediate = moe_config.intermediate_size_per_partition
        self.w13_logical_shape = (experts, 2 * intermediate, hidden)
        self.w2_logical_shape = (experts, hidden, intermediate)
        self._device = torch.device(moe_config.device)
        self._scratch_key = ("mxmoe-grouped-gemm", id(self))
        self._owned_scratch = None
        self._gemm_scratch()  # Native module and attributes ready before capture.

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        return (current_platform.is_cuda() and current_platform.has_device_capability((8, 0))
                and moe_permute_unpermute_supported())

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config):
        return _single_gpu(moe_parallel_config)

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        if weight_key is None or activation_key is not None or weight_key.scale2 is not None:
            return False
        dtype = weight_key.dtype
        return (not weight_key.symmetric and hasattr(dtype, "size_bits")
                and dtype.is_integer() and not dtype.is_signed()
                and dtype.size_bits in WEIGHT_DTYPE_BY_BITS)

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _activation_words(projection, width):
        if not projection.quantized:
            return 0
        # Packed int16 words, one FP32 scale, and int32 sums per K-group128.
        words = width * projection.packed_bits // 16 + 2 + 2 * (width // GROUP_SIZE)
        # QuantTileA issues 16-byte global loads. Each independent bank must
        # start aligned even when routed-row counts are odd.
        return (words + 7) // 8 * 8

    def _activation_workspace_columns(self):
        if self.activation_plan is None:
            return 0
        plan = self.activation_plan.plan
        hidden, intermediate = self.w13_logical_shape[2], self.w2_logical_shape[2]
        gate = self._activation_words(plan.gate, hidden)
        up = 0 if plan.gate_up_shared else self._activation_words(plan.up, hidden)
        # Down preparation reuses Gate/Up scratch after both GEMMs have finished.
        return max(gate + up, self._activation_words(plan.down, intermediate))

    @staticmethod
    def _activation_buffers(cache, rows, width, projection, start):
        flat = cache.view(-1)
        if not projection.quantized:
            empty = flat[:0]
            return empty.view(torch.int16), empty.view(torch.float32), empty.view(torch.int32)
        packed_words = width * projection.packed_bits // 16
        packed_end = start + rows * packed_words
        scale_end = packed_end + rows * 2
        sum_end = scale_end + rows * (width // GROUP_SIZE) * 2
        return (
            flat[start:packed_end].view(torch.int16).view(rows, packed_words),
            flat[packed_end:scale_end].view(torch.float32),
            flat[scale_end:sum_end].view(torch.int32).view(rows, width // GROUP_SIZE),
        )

    def _mixed_gemm(self, inputs, weights, scales, offsets, buffers, projection,
                    output, scratch, start, count):
        plan = self.activation_plan
        ops.grouped_gemm_mixed_out(
            inputs, weights, scales, offsets, self.num_bits, plan.bits[projection],
            *buffers, output, scratch, start, count, plan.plan.projections[projection].mask,
        )

    def _prepare_activation(self, projection, inputs, offsets, output, cache, start, *, silu=False):
        device_plan = self.activation_plan
        scheme = device_plan.plan.projections[projection]
        buffers = self._activation_buffers(cache, *output.shape, scheme, start)
        if scheme.quantized:
            ops.quantize_activations_out(
                inputs, offsets, device_plan.bits[projection], device_plan.clips[projection],
                *buffers, output, silu,
            )
        elif silu:
            self.activation(MoEActivation.SILU, output, inputs)
        return buffers

    def _apply_mixed(self, permuted, offsets, w1, w2, gemm1_out, act_out, gemm2_out,
                     workspace13, scratch):
        plan = self.activation_plan.plan
        rows, hidden = permuted.shape
        intermediate = self.w2_logical_shape[2]
        start = rows * max(hidden, intermediate)
        gate = self._prepare_activation(0, permuted, offsets, permuted, workspace13, start)
        if plan.gate_up_shared:
            up = gate
        else:
            up_start = start + rows * self._activation_words(plan.gate, hidden)
            up = self._prepare_activation(1, permuted, offsets, permuted, workspace13, up_start)
        # Weight storage is projection-part aligned. Identical Gate/Up schemes
        # reuse one quantized input but still issue two range-aligned GEMMs.
        self._mixed_gemm(permuted, w1, self.w1_scale, offsets, gate, 0,
                         gemm1_out, scratch, 0, intermediate)
        self._mixed_gemm(permuted, w1, self.w1_scale, offsets, up, 1,
                         gemm1_out, scratch, intermediate, intermediate)
        down = self._prepare_activation(2, gemm1_out, offsets, act_out, workspace13, start, silu=True)
        self._mixed_gemm(act_out, w2, self.w2_scale, offsets, down, 2,
                         gemm2_out, scratch, 0, hidden)

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return w1.shape[0], a1.shape[0], self.w13_logical_shape[1], a1.shape[1], topk_ids.shape[1]

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        hidden, intermediate = self.w13_logical_shape[2], self.w2_logical_shape[2]
        return ((M * topk, max(hidden, intermediate) + self._activation_workspace_columns()),
                (M * topk, max(2 * intermediate, hidden)), (M, hidden))

    def _new_scratch(self):
        return ops.workspace(self.moe_config.num_local_experts, self._device)

    def _gemm_scratch(self):
        if is_workspace_manager_initialized():
            return current_workspace_manager().get_persistent_resource(self._scratch_key, self._new_scratch)
        if self._owned_scratch is None:
            self._owned_scratch = self._new_scratch()
        return self._owned_scratch

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
              workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if expert_map is not None or apply_router_weight_on_input:
            raise ValueError("mxmoe does not support expert maps or router-weight-on-input")
        if a1q_scale is not None or a2_scale is not None or activation != MoEActivation.SILU:
            raise ValueError("mxmoe requires unquantized activations and SiLU")
        if topk_weights.dtype != torch.float32:
            raise ValueError("mxmoe unpermute requires float32 routing weights")
        tokens, hidden = hidden_states.shape
        routed, intermediate = tokens * topk_ids.shape[1], self.w2_logical_shape[2]
        # workspace13 also aliases output. Adjacent kernels must use different
        # regions; GEMM2 cannot overwrite the input that unpermute is reading.
        permuted = _resize_cache(workspace13, (routed, hidden))
        gemm1_out = _resize_cache(workspace2, (routed, 2 * intermediate))
        act_out = _resize_cache(workspace13, (routed, intermediate))
        gemm2_out = _resize_cache(workspace2, (routed, hidden))
        permute_scratch = get_moe_permute_scratch(
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts, device=self._device,
        )
        _, _, offsets, inv_perm, _ = moe_permute(
            hidden_states, None, topk_ids, self.moe_config.num_experts,
            n_local_expert=self.moe_config.num_local_experts,
            permuted_hidden_states=permuted, scratch=permute_scratch,
        )
        scratch = self._gemm_scratch()
        if self.activation_plan is None:
            ops.grouped_gemm_out(permuted, w1, self.w1_scale, offsets, self.num_bits, gemm1_out, scratch)
            self.activation(activation, act_out, gemm1_out)
            ops.grouped_gemm_out(act_out, w2, self.w2_scale, offsets, self.num_bits, gemm2_out, scratch)
        else:
            self._apply_mixed(
                permuted, offsets, w1, w2, gemm1_out, act_out, gemm2_out, workspace13, scratch,
            )
        moe_unpermute(output, gemm2_out, topk_weights, inv_perm, offsets)
