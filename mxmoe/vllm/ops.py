"""Native MxMoE grouped GEMM for the vLLM experts adapter.

Public entry points:
  prepare_weights(qweight, scales, qzeros, bits) -> (packed, scale_zero)
  workspace(experts, device) -> uint8 scratch tensor
  grouped_gemm_out(inputs, packed, scale_zero, offsets, bits, output, scratch) -> None
  prepare_weights_mixed(..., activation_bits) -> (packed, scale_zero)
  quantize_activations_out(...) -> None
  grouped_gemm_mixed_out(...) -> None

The CUDA extension is compiled on the first prepare_weights or workspace call,
not on the first grouped_gemm_out. grouped_gemm_out refuses to compile or
allocate. It launches on the current CUDA stream of the tensor device and
reads expert offsets from device memory.


Supported shape: uniform asymmetric W2/W4/W8 group128 checkpoint weights,
FP16 interface/output, optional per-expert/projection A4/A8/A16, and SM80+.
W2/W4 support A4/A8; W8 supports A8. K and N must already be multiples of
128. M may be 0 or a non-multiple of 64. Offsets are GPU int32 or int64 [E+1].
Mixed packed storage is projection-segmented and permits centered S8 expansion;
scale-zero storage retains the raw integer zero.
Reused tiles index offsets in signed int, so these products must fit in int32:
max(M, 64)*K, max(M, 64)*N, and (N/(16/bits))*K. Larger spans are rejected.

Scratch from workspace() is one in-flight slot. Same-stream GEMM1 then GEMM2
may reuse it. Overlapping slots need separate workspace() tensors. The forward
path does not cudaMalloc, copy offsets to the host, or synchronize.

The mixed path accepts independent per-expert A4/A8/A16 schemes. A4/A8 uses
symmetric per-token quantization, high-first int16 packing, FP32 scales and
group128 integer sums. It consumes exact centered checkpoint codes with raw-zero
correction in integer MMA (S4 when W<=4 and A4, S8 otherwise). A16 retains the
legacy layout/math. Projection column ranges permit different Gate/Up inputs.
All activation buffers are views into the vLLM execution-slot workspace.

Profiler kernel names include mxmoe_grouped_w{2,4,8}a16_g128_asym,
mxmoe_grouped_w{2,4}a4_g128_asym, mxmoe_grouped_w{2,4,8}a8_g128_asym,
and mxmoe_quantize_activations.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

SUPPORTED_BITS = (2, 4, 8)

_EXT = None
_ARCH = None
_LOCK = threading.Lock()
_INCLUDE = Path(__file__).resolve().parents[1] / "kernels" / "src" / "include"
_SOURCE = Path(__file__).with_name("kernels.cu")


def _as_device(device) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(f"mxmoe native ops require a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _load(device):
    global _EXT, _ARCH
    device = _as_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError(f"mxmoe grouped GEMM requires SM80+ tensor cores, got sm_{major}{minor}")
    if _EXT is not None:
        if (major, minor) != _ARCH:
            raise RuntimeError(
                f"mxmoe extension was compiled for sm_{_ARCH[0]}{_ARCH[1]} and cannot serve sm_{major}{minor}"
            )
        return _EXT
    with _LOCK:
        if _EXT is not None:
            if (major, minor) != _ARCH:
                raise RuntimeError(
                    f"mxmoe extension was compiled for sm_{_ARCH[0]}{_ARCH[1]} and cannot serve sm_{major}{minor}"
                )
            return _EXT
        if not _SOURCE.is_file():
            raise RuntimeError(f"missing native source {_SOURCE}")
        if not _INCLUDE.is_dir():
            raise RuntimeError(f"missing MxMoE include dir {_INCLUDE}")
        from torch.utils.cpp_extension import load

        arch = major * 10 + minor
        flags = [
            f"-DCOMPUTE_ARCH={arch}",
            "-DMXMOE_DEVICE_ONLY",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-std=c++17",
            "-lineinfo",
            "-O3",
            # After Torch's -D__CUDA_NO_HALF_*__ injections. See COMMON_NVCC_FLAGS.
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ]
        old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            _EXT = load(
                name=f"mxmoe_vllm_native_sm{major}{minor}",
                sources=[str(_SOURCE)],
                extra_include_paths=[str(_INCLUDE)],
                extra_cuda_cflags=flags,
                extra_cflags=["-O3"],
                with_cuda=True,
                verbose=False,
            )
        finally:
            if old_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        _ARCH = (major, minor)
        return _EXT


def prepare_weights(qweight, scales, qzeros, bits):
    """GPU-repack CT int codes. Does not requantize and does not synchronize."""
    if not torch.is_tensor(qweight) or not qweight.is_cuda:
        raise RuntimeError("prepare_weights requires CUDA qweight")
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise RuntimeError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    return _load(qweight.device).prepare_weights(qweight, scales, qzeros, bits)


def workspace(experts, device):
    """One routing-independent uint8 scratch slot. Pass it through unchanged."""
    experts = int(experts)
    if experts < 0:
        raise RuntimeError("experts must be >= 0")
    device = _as_device(device)
    ext = _load(device)
    ext.ensure_ready(device.index)
    nbytes = int(ext.workspace_bytes(experts))
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


def grouped_gemm_out(inputs, packed, scale_zero, offsets, bits, output, scratch):
    """Current-stream grouped GEMM into output. No allocation and no host offset read."""
    if _EXT is None:
        raise RuntimeError(
            "mxmoe native extension is not loaded; call prepare_weights or workspace before forward or graph capture"
        )
    _EXT.grouped_gemm_out(inputs, packed, scale_zero, offsets, int(bits), output, scratch)


def prepare_weights_mixed(qweight, scales, qzeros, bits, activation_bits):
    """Choose the exact packed layout once for each expert/projection."""
    if not torch.is_tensor(qweight) or not qweight.is_cuda:
        raise RuntimeError("prepare_weights_mixed requires CUDA qweight")
    return _load(qweight.device).prepare_weights_mixed(
        qweight, scales, qzeros, int(bits), activation_bits,
    )


def quantize_activations_out(inputs, offsets, activation_bits, clips,
                             packed, scales, sums, float_output, silu=False):
    """GPU dynamic quantization, optionally fused SiLU*Up; no forward allocation."""
    if _EXT is None:
        raise RuntimeError("initialize MxMoE weights or workspace before input preparation")
    _EXT.quantize_activations_out(
        inputs, offsets, activation_bits, clips, packed, scales, sums, float_output, silu,
    )


def grouped_gemm_mixed_out(inputs, packed, scale_zero, offsets, bits, activation_bits,
                           packed_input, activation_scales, activation_sums, output,
                           scratch, column_start, column_count, activation_mask):
    """Real integer/A16 GEMMs into a projection slice; all routing stays on GPU."""
    if _EXT is None:
        raise RuntimeError("initialize MxMoE weights or workspace before mixed GEMM")
    _EXT.grouped_gemm_mixed_out(
        inputs, packed, scale_zero, offsets, int(bits), activation_bits, packed_input,
        activation_scales, activation_sums, output, scratch, column_start, column_count,
        activation_mask,
    )
