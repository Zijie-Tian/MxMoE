"""GPTQ and compressed-tensors weight packing.

``pack_linear_state`` writes the historical qweight/qzeros/scales/g_idx layout
used by ``--export-format gptq_pack``. That layout is self-consistent with
``dequant_weight``; it is not the compressed-tensors container.

``quant_state_to_compressed_tensors`` writes the vLLM/Humming pack-quantized
container from the same ``fasterquant(return_quant_state=True)`` values:

  - weight_packed: int32 [N, K / (32/bits)], unsigned codes packed along K
  - weight_scale: float32 [N, G], the GPTQ scale with no dtype cast
  - weight_zero_point: int32 [N / (32/bits), G], raw zeros packed along N
  - weight_shape: int64 [2] = [N, K]

Codes and zeros are the GPTQ unsigned values in ``[0, 2^bits - 1]``. Zeros are
not stored as the legacy GPTQ-v1 ``zero - 1``, and they are not recentered by
the compressed-tensors signed offset. A numerical zero is ``q == z``.
Bit widths are 2, 4, and 8. Other widths are rejected.
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


COMPRESSED_TENSORS_BITS = (2, 4, 8)


def compressed_tensors_pack_factor(bits: int) -> int:
    """Values stored in one int32. Only widths that divide 32 are accepted."""
    if bits not in COMPRESSED_TENSORS_BITS:
        raise NotImplementedError(
            "compressed-tensors export supports bit widths "
            f"{COMPRESSED_TENSORS_BITS} (unsigned int32 pack, no signed offset); "
            f"got {bits}"
        )
    return 32 // bits


def compressed_tensors_num_groups(in_features: int, groupsize: int) -> int:
    if groupsize is None or groupsize <= 0:
        return 1
    return (in_features + groupsize - 1) // groupsize


def compressed_tensors_linear_shapes(
    out_features: int,
    in_features: int,
    bits: int,
    groupsize: int,
    *,
    symmetric: bool,
) -> dict[str, tuple[int, ...]]:
    """Checkpoint shapes for one packed linear. Requires aligned K and N."""
    pack_factor = compressed_tensors_pack_factor(bits)
    if in_features % pack_factor != 0:
        raise ValueError(
            f"K={in_features} is not divisible by pack factor {pack_factor}"
        )
    if not symmetric and out_features % pack_factor != 0:
        raise ValueError(
            f"N={out_features} is not divisible by pack factor {pack_factor}"
        )
    if groupsize is not None and groupsize > 0 and in_features % groupsize != 0:
        raise ValueError(
            f"K={in_features} is not divisible by group size {groupsize}"
        )
    n_groups = compressed_tensors_num_groups(in_features, groupsize)
    shapes: dict[str, tuple[int, ...]] = {
        "weight_packed": (out_features, in_features // pack_factor),
        "weight_scale": (out_features, n_groups),
        "weight_shape": (2,),
    }
    if not symmetric:
        shapes["weight_zero_point"] = (out_features // pack_factor, n_groups)
    return shapes


def _pack_unsigned_last_dim(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned codes along the last dimension, low bits first.

    ``values`` must already be a multiple of the pack factor. The first code in
    each group of ``32 // bits`` occupies bits ``[0, bits)``. No signed offset
    is added: vLLM unpacks these bits as raw unsigned codes.
    """
    pack_factor = compressed_tensors_pack_factor(bits)
    if values.shape[-1] % pack_factor != 0:
        raise ValueError(
            f"last dim {values.shape[-1]} is not divisible by pack factor {pack_factor}"
        )
    flat = values.detach().cpu().contiguous().to(torch.int64).numpy()
    lead = flat.shape[:-1]
    length = flat.shape[-1]
    rows = int(np.prod(lead)) if lead else 1
    matrix = flat.reshape(rows, length).astype(np.uint32, copy=False)
    packed = np.zeros((rows, length // pack_factor), dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    for i in range(pack_factor):
        packed |= (matrix[:, i::pack_factor] & mask) << np.uint32(bits * i)
    packed = packed.reshape(*lead, length // pack_factor)
    return torch.from_numpy(packed.view(np.int32).copy())


def _unpack_unsigned_last_dim(
    packed: torch.Tensor,
    bits: int,
    unpacked_length: int,
) -> torch.Tensor:
    """Inverse of ``_pack_unsigned_last_dim`` for a known unpacked length."""
    pack_factor = compressed_tensors_pack_factor(bits)
    flat = packed.detach().cpu().contiguous().numpy()
    lead = flat.shape[:-1]
    rows = int(np.prod(lead)) if lead else 1
    matrix = flat.reshape(rows, flat.shape[-1]).view(np.uint32)
    unpacked = np.zeros((rows, matrix.shape[1] * pack_factor), dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (matrix >> np.uint32(bits * i)) & mask
    unpacked = unpacked[:, :unpacked_length]
    return torch.from_numpy(unpacked.astype(np.int32).reshape(*lead, unpacked_length).copy())


def integer_codes_from_quant_state(
    quant_state: dict[str, torch.Tensor],
    bits: int,
) -> dict[str, torch.Tensor]:
    """Recover unsigned GPTQ codes from ``fasterquant`` scale/zero/weight.

    ``weight`` is the fake-quant value ``scale * (q - zero)``. This does not
    run GPTQ again. The incomplete last group, if any, is read through
    ``g_idx`` rather than assuming ``K`` is a multiple of the group size.
    """
    compressed_tensors_pack_factor(bits)
    weight = quant_state["weight"].detach().to(dtype=torch.float32, device="cpu")
    scale = quant_state["scale"].detach().to(dtype=torch.float32, device="cpu")
    zero = quant_state["zero"].detach().to(dtype=torch.float32, device="cpu")
    g_idx = quant_state["g_idx"].detach().to(dtype=torch.int64, device="cpu")
    out_features, in_features = weight.shape
    if scale.ndim != 2 or scale.shape[0] != out_features:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match N={out_features}"
        )
    if zero.shape != scale.shape:
        raise ValueError(
            f"zero shape {tuple(zero.shape)} != scale shape {tuple(scale.shape)}"
        )
    if g_idx.numel() != in_features:
        raise ValueError(
            f"g_idx length {g_idx.numel()} != K={in_features}"
        )
    if int(g_idx.min()) < 0 or int(g_idx.max()) >= scale.shape[1]:
        raise ValueError(
            f"g_idx range [{int(g_idx.min())}, {int(g_idx.max())}] "
            f"outside scale groups {scale.shape[1]}"
        )
    maxq = (1 << bits) - 1
    scale_k = scale.index_select(1, g_idx)
    zero_k = zero.index_select(1, g_idx)
    if torch.any(scale_k == 0):
        raise ValueError("zero scale; refusing to invent integer codes")
    codes = torch.round(weight / scale_k + zero_k).clamp(0, maxq).to(torch.int32)
    zeros = torch.round(zero).clamp(0, maxq).to(torch.int32)
    return {
        "qcodes": codes.contiguous(),
        "scale": scale.contiguous(),
        "zero": zeros.contiguous(),
        "g_idx": g_idx.to(torch.int32).contiguous(),
    }


def pad_quant_state_for_group_alignment(
    quant_state: dict[str, torch.Tensor],
    *,
    kind: str,
    pad: int,
    bits: int,
    groupsize: int,
) -> dict[str, torch.Tensor]:
    """Return a new quant state with dense-FFN group-alignment padding.

    ``kind="output_rows"`` appends ``pad`` zero rows to gate/up. Each new row
    copies the last row's scale and zero, and its codes are that zero. The
    rows are not concatenated onto a merged gate+up tensor.

    ``kind="input_cols"`` appends ``pad`` zero columns to down. Those columns
    must exactly fill the existing incomplete last group. Scale groups are not
    added, and the new codes are that group's zero rather than packed zeros.

    The input dict is not modified.
    """
    if kind not in ("output_rows", "input_cols"):
        raise ValueError(f"unsupported padding kind {kind!r}")
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    recovered = integer_codes_from_quant_state(quant_state, bits)
    weight = quant_state["weight"].detach().to(dtype=torch.float32, device="cpu").contiguous()
    scale = recovered["scale"]
    zero = quant_state["zero"].detach().to(dtype=torch.float32, device="cpu").contiguous()
    g_idx = recovered["g_idx"]
    codes = recovered["qcodes"]
    if pad == 0:
        return {
            "weight": weight,
            "scale": scale,
            "zero": zero,
            "g_idx": g_idx,
        }

    out_features, in_features = weight.shape
    if kind == "output_rows":
        new_weight = torch.zeros((pad, in_features), dtype=torch.float32)
        new_scale = scale[-1:].expand(pad, -1).clone()
        new_zero = zero[-1:].expand(pad, -1).clone()
        padded = {
            "weight": torch.cat([weight, new_weight], dim=0).contiguous(),
            "scale": torch.cat([scale, new_scale], dim=0).contiguous(),
            "zero": torch.cat([zero, new_zero], dim=0).contiguous(),
            "g_idx": g_idx.clone(),
        }
    else:
        if groupsize is None or groupsize <= 0:
            raise ValueError("input-column padding requires groupsize > 0")
        last_group = int(g_idx[-1].item())
        if last_group != scale.shape[1] - 1:
            raise ValueError(
                "g_idx tail group "
                f"{last_group} is not the last scale group {scale.shape[1] - 1}"
            )
        tail = int((g_idx == last_group).sum().item())
        if tail <= 0 or tail > groupsize:
            raise ValueError(f"invalid group tail length {tail} for group size {groupsize}")
        if not torch.equal(g_idx[-tail:], torch.full((tail,), last_group, dtype=g_idx.dtype)):
            raise ValueError(
                "incomplete group is not a contiguous input-column suffix; "
                "refusing to pad before that tail is explicit"
            )
        if tail + pad != groupsize:
            raise ValueError(
                f"refusing to pad {pad} input columns into a tail of {tail}: "
                f"that would not fill group size {groupsize} without adding a "
                "scale group or leaving the group short"
            )
        new_weight = torch.zeros((out_features, pad), dtype=torch.float32)
        new_g = torch.full((pad,), last_group, dtype=torch.int32)
        padded = {
            "weight": torch.cat([weight, new_weight], dim=1).contiguous(),
            "scale": scale.clone(),
            "zero": zero.clone(),
            "g_idx": torch.cat([g_idx, new_g], dim=0).contiguous(),
        }

    checked = integer_codes_from_quant_state(padded, bits)
    if kind == "output_rows":
        if not torch.equal(checked["qcodes"][:out_features], codes):
            raise RuntimeError("output-row padding changed original qcodes")
        if not torch.equal(checked["scale"][:out_features], scale):
            raise RuntimeError("output-row padding changed original scales")
        if not torch.equal(checked["zero"][:out_features], recovered["zero"]):
            raise RuntimeError("output-row padding changed original zeros")
        new_codes = checked["qcodes"][out_features:]
        new_zeros = checked["zero"][out_features:]
        expected = new_zeros.index_select(1, checked["g_idx"].to(torch.int64))
        if not torch.equal(new_codes, expected):
            raise RuntimeError("padded output rows are not encoded as q=z")
    else:
        if not torch.equal(checked["qcodes"][:, :in_features], codes):
            raise RuntimeError("input-column padding changed original qcodes")
        if checked["scale"].shape != scale.shape or not torch.equal(checked["scale"], scale):
            raise RuntimeError("input-column padding changed scale groups")
        if not torch.equal(checked["zero"], recovered["zero"]):
            raise RuntimeError("input-column padding changed zeros")
        tail_zero = checked["zero"][:, last_group]
        padded_codes = checked["qcodes"][:, in_features:]
        if not torch.equal(padded_codes, tail_zero.unsqueeze(1).expand_as(padded_codes)):
            raise RuntimeError("padded input columns are not encoded as q=z")
    return padded


def quant_state_to_compressed_tensors(
    quant_state: dict[str, torch.Tensor],
    bits: int,
    *,
    symmetric: bool = False,
    groupsize: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pack one ``fasterquant`` state into compressed-tensors tensors.

    ``groupsize`` is used only to check that static groups agree with ``g_idx``.
    Pass the export group size when it is known. Scales stay float32, the dtype
    ``fasterquant`` returns. vLLM casts them into the parameter dtype at load.
    """
    recovered = integer_codes_from_quant_state(quant_state, bits)
    codes = recovered["qcodes"]
    scale = recovered["scale"]
    if scale.dtype != torch.float32:
        raise RuntimeError(f"GPTQ scales must stay float32, got {scale.dtype}")
    zeros = recovered["zero"]
    g_idx = recovered["g_idx"]
    out_features, in_features = codes.shape
    n_groups = scale.shape[1]
    if groupsize is not None and groupsize > 0:
        expected_groups = compressed_tensors_num_groups(in_features, groupsize)
        if n_groups != expected_groups:
            raise ValueError(
                f"quant state has {n_groups} groups, expected {expected_groups} "
                f"for K={in_features} group size {groupsize}"
            )
        # Static groups: every column's group is min(i // groupsize, last).
        expected_g = torch.arange(in_features, dtype=torch.int32) // groupsize
        expected_g.clamp_(max=n_groups - 1)
        if not torch.equal(g_idx, expected_g):
            raise ValueError(
                "compressed-tensors export requires static groups; g_idx is not "
                "the contiguous group-size map (act-order is not supported)"
            )
    shapes = compressed_tensors_linear_shapes(
        out_features,
        in_features,
        bits,
        groupsize if groupsize is not None else (
            in_features // n_groups if n_groups > 1 else -1
        ),
        symmetric=symmetric,
    )
    packed = {
        "weight_packed": _pack_unsigned_last_dim(codes, bits).contiguous(),
        "weight_scale": scale.contiguous(),
        "weight_shape": torch.tensor(
            [out_features, in_features], dtype=torch.int64
        ),
    }
    if not symmetric:
        packed["weight_zero_point"] = _pack_unsigned_last_dim(
            zeros.transpose(0, 1).contiguous(),
            bits,
        ).transpose(0, 1).contiguous()
    for name, shape in shapes.items():
        if tuple(packed[name].shape) != shape:
            raise RuntimeError(
                f"{name} shape {tuple(packed[name].shape)} != expected {shape}"
            )
    return packed


def unpack_compressed_tensors_linear(
    packed: dict[str, torch.Tensor],
    bits: int,
    *,
    symmetric: bool = False,
) -> dict[str, torch.Tensor]:
    """Unpack one compressed-tensors linear back to codes, scales, and zeros.

    Independent of ``quant_state_to_compressed_tensors``: it reads only the
    packed tensors and ``weight_shape``. Symmetric checkpoints have no zeros;
    the returned zero tensor is empty with shape ``[N, 0]``.
    """
    shape = packed["weight_shape"].detach().cpu().to(torch.int64)
    if shape.numel() != 2:
        raise ValueError(f"weight_shape must have 2 elements, got {tuple(shape.shape)}")
    out_features = int(shape[0])
    in_features = int(shape[1])
    codes = _unpack_unsigned_last_dim(packed["weight_packed"], bits, in_features)
    if codes.shape[0] != out_features:
        raise ValueError(
            f"unpacked N={codes.shape[0]} != weight_shape N={out_features}"
        )
    scale = packed["weight_scale"].detach().cpu()
    if symmetric:
        zeros = torch.empty((out_features, 0), dtype=torch.int32)
    else:
        zp = packed["weight_zero_point"]
        zeros = _unpack_unsigned_last_dim(
            zp.transpose(0, 1).contiguous(),
            bits,
            out_features,
        ).transpose(0, 1).contiguous()
    return {
        "qcodes": codes.contiguous(),
        "scale": scale.contiguous(),
        "zero": zeros.contiguous(),
        "weight_shape": shape.contiguous(),
    }


def validate_compressed_tensors_conversion(
    quant_state: dict[str, torch.Tensor],
    bits: int,
    *,
    symmetric: bool = False,
    groupsize: int | None = None,
    original: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Pack and unpack one state. Raises if codes, scales, or zeros change.

    Scales are compared to the float32 GPTQ source, not to a narrowed copy.
    When ``original`` is the pre-padding state, also checks that old
    coordinates are unchanged and every newly added code equals its zero.
    """
    recovered = integer_codes_from_quant_state(quant_state, bits)
    source_scale = quant_state["scale"].detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.equal(recovered["scale"], source_scale):
        raise AssertionError("scale recovery changed FP32 GPTQ scales")
    packed = quant_state_to_compressed_tensors(
        quant_state,
        bits,
        symmetric=symmetric,
        groupsize=groupsize,
    )
    if packed["weight_scale"].dtype != torch.float32:
        raise AssertionError(
            f"weight_scale dtype {packed['weight_scale'].dtype} is not float32"
        )
    unpacked = unpack_compressed_tensors_linear(packed, bits, symmetric=symmetric)
    if not torch.equal(unpacked["qcodes"], recovered["qcodes"]):
        raise AssertionError("packed qcodes do not round-trip")
    if not torch.equal(unpacked["scale"], source_scale):
        raise AssertionError("packed scales do not preserve FP32 source scales")
    if not symmetric and not torch.equal(unpacked["zero"], recovered["zero"]):
        raise AssertionError("packed raw zeros do not round-trip")
    report: dict[str, Any] = {
        "bits": bits,
        "symmetric": symmetric,
        "shapes": {name: list(tensor.shape) for name, tensor in packed.items()},
        "weight_scale_dtype": "float32",
        "qcode_match": True,
        "scale_match": True,
        "zero_match": not symmetric,
    }
    if original is None:
        return report
    old = integer_codes_from_quant_state(original, bits)
    old_scale = original["scale"].detach().to(device="cpu", dtype=torch.float32).contiguous()
    old_n, old_k = old["qcodes"].shape
    new_codes = unpacked["qcodes"]
    if new_codes.shape[0] < old_n or new_codes.shape[1] < old_k:
        raise AssertionError("padding shrank the quant state")
    if not torch.equal(new_codes[:old_n, :old_k], old["qcodes"]):
        raise AssertionError("padding changed original qcodes")
    old_groups = old_scale.shape[1]
    if unpacked["scale"].shape[1] < old_groups:
        raise AssertionError("padding dropped scale groups")
    if not torch.equal(unpacked["scale"][:old_n, :old_groups], old_scale):
        raise AssertionError("padding changed original FP32 scales")
    if not symmetric:
        if new_codes.shape[1] > old_k and unpacked["zero"].shape[1] != old["zero"].shape[1]:
            raise AssertionError("down padding added a scale group")
        if not torch.equal(
            unpacked["zero"][:old_n, : old["zero"].shape[1]],
            old["zero"],
        ):
            raise AssertionError("padding changed original zeros")
        if new_codes.shape[0] > old_n:
            extra = new_codes[old_n:]
            extra_z = unpacked["zero"][old_n:].index_select(
                1, recovered["g_idx"].to(torch.int64)
            )
            if not torch.equal(extra, extra_z):
                raise AssertionError("padded rows are not q=z")
        if new_codes.shape[1] > old_k:
            extra = new_codes[:, old_k:]
            last = int(recovered["g_idx"][-1].item())
            expected = unpacked["zero"][:, last].unsqueeze(1).expand_as(extra)
            if not torch.equal(extra, expected):
                raise AssertionError("padded columns are not q=z")
    report["original_coordinates_preserved"] = True
    report["padded_zero_encoded_as_q_equals_z"] = True
    return report


def dense_ffn_alignment(
    intermediate_size: int,
    *,
    bits: int,
    groupsize: int,
) -> dict[str, int]:
    """Next dense intermediate width that satisfies group size and packing.

    For DeepSeek-V2-Lite, ``intermediate_size=10944``, ``groupsize=128``, and
    W2, this is 11008. The 64 added columns fill the existing half group.
    """
    pack_factor = compressed_tensors_pack_factor(bits)
    if groupsize is None or groupsize <= 0:
        alignment = pack_factor
    else:
        alignment = math.lcm(groupsize, pack_factor)
    if intermediate_size < 0:
        raise ValueError(f"intermediate_size must be >= 0, got {intermediate_size}")
    padded = ((intermediate_size + alignment - 1) // alignment) * alignment
    return {
        "intermediate_size": intermediate_size,
        "padded_intermediate_size": padded,
        "pad": padded - intermediate_size,
        "alignment": alignment,
        "groupsize": int(groupsize),
        "pack_factor": pack_factor,
    }



def compressed_tensors_ignore_patterns() -> list[str]:
    """Modules that must stay dense even if a target regex is broad.

    Patterns use vLLM's ``re:`` prefix. ``re.match`` is anchored at the start,
    so each pattern allows an arbitrary module prefix. ``.gate`` is the router;
    it does not match ``gate_proj``.
    """
    return [
        "lm_head",
        "re:.*lm_head$",
        "re:.*embed_tokens.*",
        "re:.*self_attn.*",
        "re:.*indexer.*",
        "re:.*norm.*",
        "re:.*\\.gate$",
        "re:.*shared_expert_gate$",
    ]



def build_compressed_tensors_config(
    *,
    bits: int,
    group_size: int,
    sym: bool,
    ignore: list[str] | None = None,
) -> dict[str, Any]:
    """vLLM ``quantization_config`` for weight-only pack-quantized WNA16.

    ``input_activations`` stays null so the loader does not requantize
    activations. Targets are the exported MLP projections, not every Linear.
    """
    compressed_tensors_pack_factor(bits)
    if group_size is None or group_size <= 0:
        strategy = "channel"
        stored_group_size = -1
    else:
        strategy = "group"
        stored_group_size = int(group_size)
    weights: dict[str, Any] = {
        "num_bits": int(bits),
        "type": "int",
        "symmetric": bool(sym),
        "strategy": strategy,
        "group_size": stored_group_size,
        "dynamic": False,
        "actorder": None,
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": [
                    "re:.*\\.(?:gate_proj|up_proj|down_proj)$",
                    "re:.*\\.(?:w1|w2|w3)$",
                ],
                "weights": weights,
                "input_activations": None,
            }
        },
        "ignore": list(ignore) if ignore is not None else compressed_tensors_ignore_patterns(),
    }
