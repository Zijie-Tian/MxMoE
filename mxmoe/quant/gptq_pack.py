"""GPTQ weight packing utilities (AutoGPTQ / GPTQ-v1 compatible layout).

Packs per-channel (per-group) quantized Linear weights into:
  - qweight: int32 packed along the *input* feature dimension
  - qzeros:  int32 packed zero-points
  - scales:  float16/bfloat16
  - g_idx:   int32 group index per input channel

Only uniform bit-widths in {2, 3, 4, 8} are supported for packing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_PACK_BITS = (2, 3, 4, 8)


def _pack_rows_int32(int_mat: np.ndarray, bits: int) -> torch.Tensor:
    """Pack rows of uint32 values into int32 words along axis=0.

    int_mat: [rows, cols] integer codes in [0, 2^bits - 1]
    returns: [ceil(rows * bits / 32), cols] int32
    """
    assert bits in SUPPORTED_PACK_BITS
    rows, cols = int_mat.shape
    int_mat = int_mat.astype(np.uint32)
    pack_per_int = 32 // bits
    # pad rows to multiple of pack_per_int
    pad = (pack_per_int - rows % pack_per_int) % pack_per_int
    if pad:
        int_mat = np.pad(int_mat, ((0, pad), (0, 0)), constant_values=0)
        rows = int_mat.shape[0]

    packed_rows = rows // pack_per_int
    packed = np.zeros((packed_rows, cols), dtype=np.uint32)
    for j in range(pack_per_int):
        packed |= int_mat[j::pack_per_int, :] << (bits * j)
    return torch.from_numpy(packed.astype(np.int32))


def pack_linear_state(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    *,
    scale_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Pack a fake-quant (or original) weight using GPTQ scales/zeros.

    Args:
        weight: [out_features, in_features] float (typically fake-quant GPTQ result)
        scales: [out_features, n_groups]
        zeros:  [out_features, n_groups]  (GPTQ additive zero: dequant = scale * (q - zero))
        g_idx:  [in_features]
        bits:   2/3/4/8

    Returns dict with qweight, qzeros, scales, g_idx (and optional bias left to caller).
    """
    if bits not in SUPPORTED_PACK_BITS:
        raise NotImplementedError(
            f"HF GPTQ pack only supports bits in {SUPPORTED_PACK_BITS}, got {bits}"
        )

    W = weight.detach().float().cpu()
    scales = scales.detach().float().cpu()
    zeros = zeros.detach().float().cpu()
    g_idx = g_idx.detach().cpu().to(torch.int32)

    out_features, in_features = W.shape
    n_groups = scales.shape[1]
    assert scales.shape == (out_features, n_groups)
    assert zeros.shape == (out_features, n_groups)
    assert g_idx.numel() == in_features

    # Integer codes: q = round(W / scale + zero)
    # Use per-column group via g_idx
    q_int = torch.empty_like(W, dtype=torch.int32)
    maxq = (1 << bits) - 1
    for j in range(in_features):
        g = int(g_idx[j].item())
        sc = scales[:, g]
        z = zeros[:, g]
        q = torch.round(W[:, j] / sc + z).clamp(0, maxq).to(torch.int32)
        q_int[:, j] = q

    # AutoGPTQ stores qweight as packed [in_features, out_features] then pack along in-dim
    # layout: qweight shape [(infeatures // 32) * bits, outfeatures]
    q_np = q_int.t().contiguous().numpy().astype(np.uint32)  # [in, out]
    qweight = _pack_rows_int32(q_np, bits)

    # zeros packed similarly: [n_groups, out] -> pack along groups dim
    # AutoGPTQ expects zeros transposed to [in_groups, out] then pack
    zeros_i = torch.round(zeros).clamp(0, maxq).to(torch.int32).t().contiguous()  # [G, out]
    z_np = zeros_i.numpy().astype(np.uint32)
    # Pack along group dimension; for small G, pad to pack width
    qzeros = _pack_rows_int32(z_np, bits)

    return {
        "qweight": qweight.contiguous(),
        "qzeros": qzeros.contiguous(),
        "scales": scales.to(dtype=scale_dtype).contiguous(),
        "g_idx": g_idx.contiguous(),
    }


def dequant_weight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize packed GPTQ tensors back to float32 [out, in] (for round-trip checks)."""
    if bits not in SUPPORTED_PACK_BITS:
        raise NotImplementedError(bits)

    pack_per_int = 32 // bits
    maxq = (1 << bits) - 1
    # unpack qweight: [packed_in, out] -> [in, out]
    qw = qweight.cpu().numpy().astype(np.uint32)
    packed_rows, cols = qw.shape
    rows = packed_rows * pack_per_int
    unpacked = np.zeros((rows, cols), dtype=np.uint32)
    mask = (1 << bits) - 1
    for j in range(pack_per_int):
        unpacked[j::pack_per_int, :] = (qw >> (bits * j)) & mask
    unpacked = unpacked[:in_features, :]  # [in, out]
    q_int = torch.from_numpy(unpacked.astype(np.int32)).t().contiguous()  # [out, in]

    # unpack zeros: [packed_g, out]
    n_groups = scales.shape[1]
    qz = qzeros.cpu().numpy().astype(np.uint32)
    z_rows = qz.shape[0] * pack_per_int
    z_unpacked = np.zeros((z_rows, qz.shape[1]), dtype=np.uint32)
    for j in range(pack_per_int):
        z_unpacked[j::pack_per_int, :] = (qz >> (bits * j)) & mask
    z_unpacked = z_unpacked[:n_groups, :]
    zeros = torch.from_numpy(z_unpacked.astype(np.float32)).t().contiguous()  # [out, G]

    scales = scales.float().cpu()
    g_idx = g_idx.cpu().to(torch.int64)
    W = torch.empty((out_features, in_features), dtype=torch.float32)
    for j in range(in_features):
        g = int(g_idx[j].item())
        W[:, j] = scales[:, g] * (q_int[:, j].float() - zeros[:, g])
    return W


def quant_state_to_packed(
    quant_state: dict[str, torch.Tensor],
    bits: int,
    *,
    scale_dtype: torch.dtype = torch.float16,
    bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Convert ``fasterquant(..., return_quant_state=True)`` output to packed tensors."""
    packed = pack_linear_state(
        quant_state["weight"],
        quant_state["scale"],
        quant_state["zero"],
        quant_state["g_idx"],
        bits,
        scale_dtype=scale_dtype,
    )
    if bias is not None:
        packed["bias"] = bias.detach().cpu().to(dtype=scale_dtype)
    return packed


def build_quantize_config_dict(
    *,
    bits: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
    damp_percent: float = 0.01,
    true_sequential: bool = True,
    quant_method: str = "gptq",
    checkpoint_format: str = "gptq",
    modules_to_not_convert: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a quantize_config / HF quantization_config style dict."""
    cfg = {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "desc_act": desc_act,
        "damp_percent": damp_percent,
        "true_sequential": true_sequential,
        "quant_method": quant_method,
        "checkpoint_format": checkpoint_format,
        "modules_in_block_to_quantize": None,
        "modules_to_not_convert": modules_to_not_convert or [],
    }
    if meta:
        cfg["meta"] = meta
    return cfg
