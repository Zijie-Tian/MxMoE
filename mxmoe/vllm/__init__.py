"""MxMoE vLLM integration.

Native ops are ``mxmoe.vllm.ops``; the extension loads on prepare_weights or
workspace. Import-safe before ``VllmConfig`` construction: no torch or vLLM.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def activation_config_digest(path: str) -> str:
    """Hash qconfig bytes so an in-place edit cannot reuse a compile cache entry."""
    resolved = Path(path).expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Cannot read mxmoe_activation_config {str(resolved)!r}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_kernel_selection(kernel_config, vllm_config) -> None:
    """Validate deployment policy before the fused-MoE selector runs."""
    if kernel_config.moe_backend != "mxmoe":
        if kernel_config.mxmoe_activation_config is not None:
            raise ValueError("mxmoe_activation_config requires moe_backend='mxmoe'")
        return
    model_config = vllm_config.model_config
    if model_config is not None and model_config.quantization != "compressed-tensors":
        raise ValueError(
            "moe_backend='mxmoe' requires compressed-tensors quantization; "
            "other quantization methods may bypass the native MoE selector."
        )
    # FusedMoEParallelConfig omits PP, so it must be checked here.
    pipeline_parallel_size = vllm_config.parallel_config.pipeline_parallel_size
    if pipeline_parallel_size != 1:
        raise ValueError(
            "moe_backend='mxmoe' requires a single GPU "
            "(pipeline_parallel_size=1); "
            f"got pipeline_parallel_size={pipeline_parallel_size}."
        )


def reject_quantized_activations(has_input_quant: bool) -> None:
    """The interface stays FP16; internal quantization uses the activation policy."""
    if has_input_quant:
        raise ValueError(
            "mxmoe requires unquantized FP16 input metadata; "
            "configure internal A4/A8 with mxmoe_activation_config"
        )
