"""Worker-side evidence that a compressed-tensors W2A16 model is really
running Humming for dense linears and routed experts.

The timing process must not install these wrappers. ``GptqAuditExtension``
only reads loaded modules. ``GptqAuditWorker`` installs the wrappers before
weights are loaded, and only the eager observation engine uses that worker.
"""

from __future__ import annotations

import re
from typing import Any

WORKER_EXTENSION = "mxmoe.vllm_gptq_audit.GptqAuditExtension"
AUDIT_WORKER = "mxmoe.vllm_gptq_audit.GptqAuditWorker"

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_DENSE_PROJECTIONS = {"gate_up_proj", "down_proj", "gate_proj", "up_proj"}
_EXCLUDED_UNQUANTIZED = {"attention", "embed", "lm_head", "router", "norm"}


def _dtype_name(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    if text.startswith("torch."):
        return text[len("torch.") :]
    return text


def _is_a16_dtype(value: Any) -> bool:
    return _dtype_name(value) in {"float16", "bfloat16"}


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _layer_index(prefix: str) -> int | None:
    match = _LAYER_INDEX.search(prefix)
    if match is None:
        return None
    return int(match.group(1))


def _leaf(prefix: str) -> str:
    return prefix.rsplit(".", 1)[-1]


def _canonical_prefix(named_prefix: str, module: Any) -> str:
    return str(
        getattr(module, "prefix", None)
        or getattr(module, "layer_name", None)
        or named_prefix
    )


class _CallBucket:
    def __init__(self) -> None:
        self.count = 0
        self.prefixes: set[str] = set()
        self.activation_dtypes: set[str] = set()
        self.kernels: set[str] = set()
        self.experts_classes: set[str] = set()
        self.sublayers: dict[str, int] = {}
        self.rows_min: int | None = None
        self.rows_max: int | None = None
        self.floating_weight_calls = 0
        self.phase_sources: set[str] = set()

    def add(
        self,
        *,
        prefix: str,
        rows: int,
        activation_dtype: str,
        kernel: str | None,
        experts_cls: str | None,
        sublayer: str | None,
        weight_is_floating: bool,
        phase_source: str,
    ) -> None:
        self.count += 1
        if prefix:
            self.prefixes.add(prefix)
        self.activation_dtypes.add(activation_dtype)
        if kernel:
            self.kernels.add(kernel)
        if experts_cls:
            self.experts_classes.add(experts_cls)
        if sublayer:
            self.sublayers[sublayer] = self.sublayers.get(sublayer, 0) + 1
        self.rows_min = rows if self.rows_min is None else min(self.rows_min, rows)
        self.rows_max = rows if self.rows_max is None else max(self.rows_max, rows)
        if weight_is_floating:
            self.floating_weight_calls += 1
        self.phase_sources.add(phase_source)

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "prefixes": sorted(self.prefixes),
            "activation_dtypes": sorted(self.activation_dtypes),
            "kernels": sorted(self.kernels),
            "experts_classes": sorted(self.experts_classes),
            "sublayers": dict(sorted(self.sublayers.items())),
            "rows_min": self.rows_min,
            "rows_max": self.rows_max,
            "floating_weight_calls": self.floating_weight_calls,
            "phase_sources": sorted(self.phase_sources),
        }


class _Recorder:
    def __init__(self) -> None:
        self.installed = False
        self.enabled = False
        self.batch_size = 1
        self.prompt_len = 1
        self.token_phase = "unscoped"
        self.required_dense_prefixes: list[str] = []
        self.required_moe_prefixes: list[str] = []
        self.original_dense = None
        self.original_moe = None
        self.forward_tags = 0
        self.calls: dict[str, dict[str, _CallBucket]] = {}
        self.reset()

    def reset(self) -> None:
        self.calls = {
            phase: {"dense": _CallBucket(), "moe": _CallBucket()}
            for phase in ("prefill", "decode", "unscoped")
        }
        self.forward_tags = 0
        self.token_phase = "unscoped"

    def begin(self, batch_size: int, prompt_len: int) -> None:
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.reset()
        self.enabled = True

    def snapshot(self) -> dict[str, Any]:
        return {
            phase: {kind: bucket.snapshot() for kind, bucket in kinds.items()}
            for phase, kinds in self.calls.items()
        }


_RECORDER = _Recorder()


def _iter_metadata(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_metadata(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_metadata(item)
        return
    yield value


def _phase_from_metadata() -> str:
    try:
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
    except Exception:
        return "unscoped"
    for item in _iter_metadata(metadata):
        num_prefills = getattr(item, "num_prefills", None)
        num_decode_tokens = getattr(item, "num_decode_tokens", None)
        max_query_len = getattr(item, "max_query_len", None)
        if num_prefills == 0 and (num_decode_tokens or 0) > 0:
            return "decode"
        if (num_prefills or 0) > 0 and (num_decode_tokens or 0) == 0:
            return "prefill"
        if max_query_len == 1 and (num_prefills or 0) == 0:
            return "decode"
        if (
            max_query_len is not None
            and max_query_len > 1
            and (num_decode_tokens or 0) == 0
        ):
            return "prefill"
    return "unscoped"


def _current_phase() -> tuple[str, str]:
    metadata_phase = _phase_from_metadata()
    if metadata_phase != "unscoped":
        return metadata_phase, "attn_metadata"
    if _RECORDER.token_phase != "unscoped":
        return _RECORDER.token_phase, "model_forward_token_count"
    return "unscoped", "none"


def _token_count(args: tuple, kwargs: dict) -> int | None:
    import torch

    positions = kwargs.get("positions")
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if positions is None and len(args) > 1:
        positions = args[1]
    tensor = positions if torch.is_tensor(positions) else input_ids
    if not torch.is_tensor(tensor):
        return None
    if tensor.ndim >= 2 and tensor.shape[0] <= 4 and tensor.shape[-1] > tensor.shape[0]:
        return int(tensor.shape[-1])
    return int(tensor.shape[0])


def _model_pre_hook(_module, args, kwargs) -> None:
    if not _RECORDER.enabled:
        return
    count = _token_count(args, kwargs)
    _RECORDER.forward_tags += 1
    if count is None:
        _RECORDER.token_phase = "unscoped"
    elif count <= _RECORDER.batch_size:
        _RECORDER.token_phase = "decode"
    else:
        _RECORDER.token_phase = "prefill"


def _note_dense(layer, x) -> None:
    if not _RECORDER.enabled:
        return
    phase, source = _current_phase()
    rows = int(x.reshape(-1, x.shape[-1]).shape[0])
    weight = getattr(layer, "weight", None)
    scheme = _linear_scheme(layer)
    kernel = getattr(scheme, "kernel", None)
    if kernel is None:
        kernel = getattr(_method(layer), "kernel", None)
    prefix = str(
        getattr(layer, "_gptq_audit_prefix", "") or getattr(layer, "prefix", "") or ""
    )
    _RECORDER.calls[phase]["dense"].add(
        prefix=prefix,
        rows=rows,
        activation_dtype=_dtype_name(x.dtype),
        kernel=type(kernel).__name__ if kernel is not None else None,
        experts_cls=None,
        sublayer=None,
        weight_is_floating=bool(weight is not None and weight.is_floating_point()),
        phase_source=source,
    )


def _note_moe(experts, sublayer_name, inputs, weight) -> None:
    if not _RECORDER.enabled:
        return
    phase, source = _current_phase()
    rows = int(inputs.shape[0]) if getattr(inputs, "ndim", 0) else 0
    _RECORDER.calls[phase]["moe"].add(
        prefix=str(getattr(experts, "_gptq_audit_prefix", "") or ""),
        rows=rows,
        activation_dtype=_dtype_name(inputs.dtype),
        kernel="humming_forward",
        experts_cls=type(experts).__name__,
        sublayer=str(sublayer_name),
        weight_is_floating=bool(weight is not None and weight.is_floating_point()),
        phase_source=source,
    )


def install_backend_counters() -> None:
    """Wrap Humming entry points. Call this before the first forward."""
    if _RECORDER.installed:
        return
    import vllm.model_executor.layers.quantization.utils.humming as humming_pkg
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )
    from vllm.model_executor.layers.quantization.utils.humming import (
        linear as linear_mod,
    )

    original_dense = linear_mod.apply_humming_linear
    original_moe = HummingExpertsBase.humming_forward

    def apply_humming_linear(layer, x, *args, **kwargs):
        _note_dense(layer, x)
        return original_dense(layer, x, *args, **kwargs)

    def humming_forward(self, sublayer_name, inputs, weight, *args, **kwargs):
        _note_moe(self, sublayer_name, inputs, weight)
        return original_moe(self, sublayer_name, inputs, weight, *args, **kwargs)

    linear_mod.apply_humming_linear = apply_humming_linear
    humming_pkg.apply_humming_linear = apply_humming_linear
    HummingExpertsBase.humming_forward = humming_forward
    _RECORDER.original_dense = original_dense
    _RECORDER.original_moe = original_moe
    _RECORDER.installed = True


def uninstall_backend_counters() -> None:
    if not _RECORDER.installed:
        return
    import vllm.model_executor.layers.quantization.utils.humming as humming_pkg
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )
    from vllm.model_executor.layers.quantization.utils.humming import (
        linear as linear_mod,
    )

    linear_mod.apply_humming_linear = _RECORDER.original_dense
    humming_pkg.apply_humming_linear = _RECORDER.original_dense
    HummingExpertsBase.humming_forward = _RECORDER.original_moe
    _RECORDER.installed = False
    _RECORDER.enabled = False


def _role(prefix: str, module: Any) -> str:
    name = _leaf(prefix)
    class_name = type(module).__name__
    if class_name == "GateLinear" or (
        name == "gate" and ".mlp." in prefix and "gate_proj" not in name
    ):
        return "router"
    if ".self_attn." in prefix:
        return "attention"
    if name in {"embed_tokens", "embed_positions"} or prefix.endswith("embed_tokens"):
        return "embed"
    if name == "lm_head":
        return "lm_head"
    if "norm" in name.lower():
        return "norm"
    if class_name == "RoutedExperts" or name == "experts":
        return "routed_experts"
    if ".shared_experts." in prefix and name in _DENSE_PROJECTIONS:
        return "shared_expert"
    if (
        ".experts." not in prefix
        and ".shared_experts." not in prefix
        and ".self_attn." not in prefix
        and name in _DENSE_PROJECTIONS
    ):
        return "dense_mlp"
    return "other"


def _method(module: Any) -> Any:
    return getattr(module, "quant_method", None)


def _linear_scheme(module: Any) -> Any:
    """Compressed-tensors stores the WNA16 scheme on the layer, not the method.

    ``CompressedTensorsLinearMethod`` only forwards to ``layer.scheme``. The
    scheme owns ``kernel``, ``num_bits``, ``group_size``, and ``symmetric``.
    """
    scheme = getattr(module, "scheme", None)
    if scheme is None:
        return None
    if hasattr(scheme, "kernel") or hasattr(scheme, "num_bits"):
        return scheme
    return None


def _scheme_attr(scheme: Any, method: Any, name: str) -> Any:
    if scheme is not None and hasattr(scheme, name):
        return getattr(scheme, name)
    return getattr(method, name, None)


def _is_unquantized_method(method: Any) -> bool:
    if method is None:
        return True
    name = type(method).__name__
    return "Unquantized" in name or name == "_GateLinearMethod"


def _floating_weight_names(module: Any) -> list[str]:
    names = []
    for name, param in module.named_parameters(recurse=False):
        lowered = name.lower()
        if "weight" in lowered and "scale" not in lowered and param.is_floating_point():
            names.append(f"{name}:{_dtype_name(param.dtype)}")
    return names


def _humming_input_bits(obj: Any) -> int | None:
    bits = getattr(getattr(obj, "a_dtype", None), "num_bits", None)
    if bits is None:
        return None
    return int(bits)


def _stamp_moe_prefix(module: Any, prefix: str) -> None:
    method = _method(module)
    if method is None:
        return
    objects = [method]
    kernel = getattr(method, "moe_kernel", None)
    if kernel is not None:
        objects.append(kernel)
        fused = getattr(kernel, "fused_experts", None)
        if fused is not None:
            objects.append(fused)
    for obj in objects:
        if hasattr(obj, "humming_forward"):
            obj._gptq_audit_prefix = prefix


def _linear_record(prefix: str, module: Any, role: str) -> dict[str, Any]:
    method = _method(module)
    scheme = _linear_scheme(module)
    kernel = _scheme_attr(scheme, method, "kernel")
    config = getattr(kernel, "config", None)
    weight_type = getattr(config, "weight_type", None)
    bits = _scheme_attr(scheme, method, "num_bits")
    if bits is None and weight_type is not None:
        bits = getattr(weight_type, "size_bits", None)
    act = getattr(config, "act_type", None)
    if act is None:
        act = _scheme_attr(scheme, method, "input_dtype")
    layer_config = getattr(kernel, "layer_config", None)
    group_size = _scheme_attr(scheme, method, "group_size")
    symmetric = _scheme_attr(scheme, method, "symmetric")
    return {
        "prefix": prefix,
        "role": role,
        "class_name": type(module).__name__,
        "quant_method": type(method).__name__ if method is not None else None,
        "scheme": type(scheme).__name__ if scheme is not None else None,
        "kernel": type(kernel).__name__ if kernel is not None else None,
        "moe_backend": None,
        "experts_cls": None,
        "weight_bits": None if bits is None else int(bits),
        "group_size": group_size,
        "symmetric": symmetric,
        "activation_dtype": None if act is None else _dtype_name(act),
        "humming_input_bits": _humming_input_bits(layer_config),
        "floating_weights": _floating_weight_names(module),
        "static_selection": (
            "humming"
            if kernel is not None and type(kernel).__name__ == "HummingLinearKernel"
            else "unquantized"
            if _is_unquantized_method(method)
            else "other"
        ),
    }


def _moe_record(prefix: str, module: Any) -> dict[str, Any]:
    method = _method(module)
    backend = _enum_name(getattr(method, "wna16_backend", None))
    experts_cls = getattr(method, "experts_cls", None)
    kernel = getattr(method, "moe_kernel", None)
    fused = getattr(kernel, "fused_experts", None)
    configs = getattr(fused, "humming_configs", None) or {}
    input_bits = []
    for name, config in configs.items():
        bits = _humming_input_bits(config)
        if bits is not None:
            input_bits.append(f"{name}:{bits}")
    act = getattr(method, "input_dtype", None)
    return {
        "prefix": prefix,
        "role": "routed_experts",
        "class_name": type(module).__name__,
        "quant_method": type(method).__name__ if method is not None else None,
        "kernel": type(fused).__name__ if fused is not None else None,
        "moe_backend": backend,
        "experts_cls": None if experts_cls is None else experts_cls.__name__,
        "weight_bits": getattr(method, "num_bits", None),
        "group_size": getattr(method, "group_size", None),
        "symmetric": getattr(method, "symmetric", None),
        "activation_dtype": None if act is None else _dtype_name(act),
        "humming_input_bits": None if not input_bits else ",".join(input_bits),
        "floating_weights": _floating_weight_names(module),
        "static_selection": "humming" if backend == "HUMMING" else "other",
    }


def _is_moe_layer(layer_idx: int, first_k: int, moe_freq: int) -> bool:
    return layer_idx >= first_k and layer_idx % moe_freq == 0


def _projection_gap(names: set[str]) -> str | None:
    has_down = "down_proj" in names
    has_up = "gate_up_proj" in names or {"gate_proj", "up_proj"} <= names
    if has_down and has_up:
        return None
    return f"found {sorted(names) or ['none']}, expected gate/up and down"


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _engine_snapshot(worker: Any) -> dict[str, Any]:
    config = worker.vllm_config
    compilation = config.compilation_config
    return {
        "quantization": config.model_config.quantization,
        "dtype": _dtype_name(config.model_config.dtype),
        "enforce_eager": bool(config.model_config.enforce_eager),
        "tensor_parallel_size": int(config.parallel_config.tensor_parallel_size),
        "enable_prefix_caching": bool(config.cache_config.enable_prefix_caching),
        "moe_backend": config.kernel_config.moe_backend,
        "linear_backend": config.kernel_config.linear_backend,
        "compilation_mode": _enum_name(compilation.mode),
        "cudagraph_mode": _enum_name(compilation.cudagraph_mode),
        "max_num_batched_tokens": int(config.scheduler_config.max_num_batched_tokens),
        "max_num_seqs": int(config.scheduler_config.max_num_seqs),
    }


def _check_w2_linear(
    record: dict[str, Any], weight_bits: int, group_size: int, failures: list[str]
) -> None:
    prefix = record["prefix"]
    if record["static_selection"] == "unquantized" or record["floating_weights"]:
        failures.append(
            f"{prefix} is floating ({record['quant_method']}, "
            f"weights {record['floating_weights'] or 'unquantized'}); expected Humming W2A16"
        )
        return
    if record["kernel"] != "HummingLinearKernel":
        failures.append(
            f"{prefix} selected {record['kernel'] or record['quant_method']}, "
            "not HummingLinearKernel"
        )
    if record["weight_bits"] != weight_bits:
        failures.append(
            f"{prefix} weight bits are {record['weight_bits']}, expected {weight_bits}"
        )
    if _as_int(record["group_size"]) != group_size:
        failures.append(
            f"{prefix} group size is {record['group_size']}, expected {group_size}"
        )
    if record["symmetric"] is None or bool(record["symmetric"]):
        failures.append(f"{prefix} symmetric={record['symmetric']}, expected asymmetric W2")
    if not _is_a16_dtype(record["activation_dtype"]):
        failures.append(
            f"{prefix} activation dtype is {record['activation_dtype']}, "
            "expected float16 or bfloat16"
        )
    input_bits = record["humming_input_bits"]
    if input_bits is not None and input_bits < 16:
        failures.append(f"{prefix} Humming input schema is {input_bits}-bit, expected A16")


def _check_w2_moe(
    record: dict[str, Any], weight_bits: int, group_size: int, failures: list[str]
) -> None:
    prefix = record["prefix"]
    if record["floating_weights"] or record["quant_method"] in {
        None,
        "UnquantizedFusedMoEMethod",
    }:
        failures.append(
            f"{prefix} routed experts are floating ({record['quant_method']}, "
            f"weights {record['floating_weights'] or 'missing'}); expected Humming W2A16"
        )
        return
    if record["moe_backend"] != "HUMMING":
        failures.append(
            f"{prefix} fused MoE backend is {record['moe_backend']}, expected HUMMING"
        )
    experts_cls = record["experts_cls"] or ""
    if "Humming" not in experts_cls:
        failures.append(
            f"{prefix} experts class is {experts_cls or 'missing'}, expected Humming"
        )
    if _as_int(record["weight_bits"]) != weight_bits:
        failures.append(
            f"{prefix} weight bits are {record['weight_bits']}, expected {weight_bits}"
        )
    if _as_int(record["group_size"]) != group_size:
        failures.append(
            f"{prefix} group size is {record['group_size']}, expected {group_size}"
        )
    if record["symmetric"] is None or bool(record["symmetric"]):
        failures.append(f"{prefix} symmetric={record['symmetric']}, expected asymmetric W2")
    if not _is_a16_dtype(record["activation_dtype"]):
        failures.append(
            f"{prefix} configured activation dtype is {record['activation_dtype']}, expected A16"
        )
    text = record["humming_input_bits"]
    if isinstance(text, str):
        for part in text.split(","):
            bits = int(part.split(":")[-1])
            if bits < 16:
                failures.append(f"{prefix} Humming {part} input is not A16")


def inspect_loaded_model(
    worker: Any, *, weight_bits: int, group_size: int
) -> dict[str, Any]:
    """Classify loaded modules. This is static selection, not an observed call."""
    model = worker.model_runner.model
    hf = worker.vllm_config.model_config.hf_config
    first_k = int(getattr(hf, "first_k_dense_replace", 0) or 0)
    moe_freq = int(getattr(hf, "moe_layer_freq", 1) or 1)
    num_layers = int(getattr(hf, "num_hidden_layers"))
    n_shared = getattr(hf, "n_shared_experts", None)
    failures: list[str] = []
    notes: list[str] = []
    required: list[dict[str, Any]] = []
    excluded_counts = {role: 0 for role in _EXCLUDED_UNQUANTIZED}
    excluded_quantized: list[str] = []
    dense_names: dict[int, set[str]] = {}
    shared_names: dict[int, set[str]] = {}
    expert_layers: set[int] = set()

    for named_prefix, module in model.named_modules():
        role = _role(named_prefix, module)
        if role in _EXCLUDED_UNQUANTIZED:
            excluded_counts[role] += 1
            method = _method(module)
            if not _is_unquantized_method(method):
                excluded_quantized.append(f"{named_prefix}: {type(method).__name__}")
            continue
        layer_idx = _layer_index(named_prefix)
        if role == "dense_mlp":
            if layer_idx is None or layer_idx >= first_k:
                continue
            prefix = _canonical_prefix(named_prefix, module)
            module._gptq_audit_prefix = prefix
            record = _linear_record(prefix, module, role)
            required.append(record)
            dense_names.setdefault(layer_idx, set()).add(_leaf(prefix))
            _check_w2_linear(record, weight_bits, group_size, failures)
        elif role == "shared_expert":
            if layer_idx is None or not _is_moe_layer(layer_idx, first_k, moe_freq):
                continue
            prefix = _canonical_prefix(named_prefix, module)
            module._gptq_audit_prefix = prefix
            record = _linear_record(prefix, module, role)
            required.append(record)
            shared_names.setdefault(layer_idx, set()).add(_leaf(prefix))
            _check_w2_linear(record, weight_bits, group_size, failures)
        elif role == "routed_experts" and type(module).__name__ == "RoutedExperts":
            if layer_idx is None or not _is_moe_layer(layer_idx, first_k, moe_freq):
                continue
            prefix = _canonical_prefix(named_prefix, module)
            record = _moe_record(prefix, module)
            required.append(record)
            expert_layers.add(layer_idx)
            _stamp_moe_prefix(module, prefix)
            _check_w2_moe(record, weight_bits, group_size, failures)

    for layer_idx in range(first_k):
        gap = _projection_gap(dense_names.get(layer_idx, set()))
        if gap is not None:
            failures.append(
                f"layer {layer_idx} dense MLP is missing a W2 projection ({gap})"
            )
    for layer_idx in range(first_k, num_layers):
        if not _is_moe_layer(layer_idx, first_k, moe_freq):
            continue
        if layer_idx not in expert_layers:
            failures.append(
                f"layer {layer_idx} routed experts were not loaded as quantized experts"
            )
        if n_shared:
            gap = _projection_gap(shared_names.get(layer_idx, set()))
            if gap is not None:
                failures.append(
                    f"layer {layer_idx} shared experts are missing a W2 projection ({gap})"
                )

    if excluded_quantized:
        failures.append(
            "attention, embed, lm_head, router, or norm used a quantized method: "
            + "; ".join(excluded_quantized[:12])
        )
    if excluded_counts["attention"] == 0 or excluded_counts["lm_head"] == 0:
        notes.append(
            "attention or lm_head was not found by name; excluded-role counts are "
            f"{excluded_counts}"
        )

    engine = _engine_snapshot(worker)
    if engine["moe_backend"] != "humming" or engine["linear_backend"] != "humming":
        failures.append(
            "engine backends are "
            f"moe={engine['moe_backend']}, linear={engine['linear_backend']}; "
            "expected humming for both"
        )
    if engine["enable_prefix_caching"]:
        failures.append("prefix caching is enabled; repeated timing would reuse KV")
    if engine["quantization"] != "compressed-tensors":
        failures.append(
            f"quantization method is {engine['quantization']}, expected compressed-tensors"
        )

    dense_prefixes = [
        item["prefix"] for item in required if item["role"] != "routed_experts"
    ]
    moe_prefixes = [item["prefix"] for item in required if item["role"] == "routed_experts"]
    _RECORDER.required_dense_prefixes = dense_prefixes
    _RECORDER.required_moe_prefixes = moe_prefixes
    return {
        "engine": engine,
        "first_k_dense_replace": first_k,
        "moe_layer_freq": moe_freq,
        "num_hidden_layers": num_layers,
        "n_shared_experts": n_shared,
        "required_modules": required,
        "excluded_counts": excluded_counts,
        "required_dense_prefixes": dense_prefixes,
        "required_moe_prefixes": moe_prefixes,
        "failures": failures,
        "notes": notes,
        "selection_kind": "static_loaded_module",
    }


def _observed_failures() -> list[str]:
    failures = []
    calls = _RECORDER.calls
    for phase in ("prefill", "decode"):
        dense = calls[phase]["dense"]
        moe = calls[phase]["moe"]
        missing_dense = sorted(set(_RECORDER.required_dense_prefixes) - dense.prefixes)
        missing_moe = sorted(set(_RECORDER.required_moe_prefixes) - moe.prefixes)
        if _RECORDER.required_dense_prefixes and missing_dense:
            failures.append(
                f"{phase} did not call apply_humming_linear for: " + ", ".join(missing_dense)
            )
        if _RECORDER.required_moe_prefixes and missing_moe:
            failures.append(
                f"{phase} did not call Humming MoE humming_forward for: "
                + ", ".join(missing_moe)
            )
        if _RECORDER.required_moe_prefixes:
            for sublayer in ("w13", "w2"):
                if moe.sublayers.get(sublayer, 0) <= 0:
                    failures.append(f"{phase} Humming MoE did not call {sublayer}")
        for kind, bucket in (("dense", dense), ("moe", moe)):
            if bucket.floating_weight_calls:
                failures.append(
                    f"{phase} {kind} Humming call saw floating weights "
                    f"({bucket.floating_weight_calls} calls)"
                )
            bad = sorted(
                dtype for dtype in bucket.activation_dtypes if not _is_a16_dtype(dtype)
            )
            if bad:
                failures.append(
                    f"{phase} {kind} activation dtypes were {bad}, expected A16"
                )
    if calls["unscoped"]["dense"].count or calls["unscoped"]["moe"].count:
        failures.append(
            "Humming calls ran without a prefill/decode phase tag: "
            f"dense {calls['unscoped']['dense'].count}, "
            f"moe {calls['unscoped']['moe'].count}"
        )
    return failures


def inspect_humming_a4(worker) -> dict:
    """Read actual loaded kernel configurations; do not add forward hooks."""
    records, failures = [], []
    counts = {"HummingLinearMethod": 0, "HummingMoEMethod": 0}
    for prefix, layer in worker.model_runner.model.named_modules():
        leaf = prefix.rsplit(".", 1)[-1]
        if ".mlp." not in prefix or not (
            leaf in {"gate_proj", "up_proj", "gate_up_proj", "down_proj"}
            or type(layer).__name__ == "RoutedExperts"
        ):
            continue
        method = getattr(layer, "quant_method", None)
        name = type(method).__name__
        if name == "HummingLinearMethod":
            configs = {"linear": getattr(method, "layer_config", None)}
        elif name == "HummingMoEMethod":
            configs = getattr(method, "humming_configs", {})
        else:
            failures.append(f"{prefix} selected {name}, not Humming")
            continue
        counts[name] += 1
        if not configs or any(cfg is None for cfg in configs.values()):
            failures.append(f"{prefix} has no loaded Humming configuration")
            continue
        for part, cfg in configs.items():
            record = {"prefix": prefix, "part": part, "method": name,
                      "weight_bits": cfg.b_dtype.num_bits,
                      "activation_bits": cfg.a_dtype.num_bits,
                      "activation_dtype": str(cfg.a_dtype),
                      "input_quant_mode": str(cfg.input_quant_mode.value),
                      "weight_group_size": cfg.weight_scale_group_size}
            records.append(record)
            if (record["weight_bits"], record["activation_bits"]) != (2, 4):
                failures.append(f"{prefix}.{part} is not W2A4: {record}")
    hf = worker.vllm_config.model_config.hf_config
    first = int(getattr(hf, "first_k_dense_replace", 0))
    moe = sum(i >= first and i % getattr(hf, "moe_layer_freq", 1) == 0
              for i in range(hf.num_hidden_layers))
    dense = 2 * (hf.num_hidden_layers - moe + (moe if getattr(hf, "n_shared_experts", 0) else 0))
    if counts != {"HummingLinearMethod": dense, "HummingMoEMethod": moe}:
        failures.append(f"Incomplete W2A4 module coverage: {counts}")
    tp = worker.vllm_config.parallel_config.tensor_parallel_size
    if tp != 1:
        failures.append(f"Expected single-GPU execution, got TP={tp}")
    return {"counts": counts, "tensor_parallel_size": tp,
            "loaded_configs": records, "failures": failures}


class GptqAuditExtension:
    """Methods injected into the default worker. They do not wrap kernels."""

    def gptq_audit_inspect(self, weight_bits: int = 2, group_size: int = 128) -> dict:
        return inspect_loaded_model(self, weight_bits=weight_bits, group_size=group_size)

    def gptq_audit_inspect_a4(self) -> dict:
        return inspect_humming_a4(self)

    def gptq_audit_begin(self, batch_size: int, prompt_len: int) -> dict:
        install_backend_counters()
        _RECORDER.begin(batch_size, prompt_len)
        model = self.model_runner.model  # type: ignore[attr-defined]
        handles = [model.register_forward_pre_hook(_model_pre_hook, with_kwargs=True)]
        for named_prefix, module in model.named_modules():
            if type(module).__name__ != "RoutedExperts":
                continue
            _stamp_moe_prefix(module, _canonical_prefix(named_prefix, module))
        self._gptq_audit_handles = handles  # type: ignore[attr-defined]
        return {
            "recording": True,
            "batch_size": batch_size,
            "prompt_len": prompt_len,
            "wrappers_installed": _RECORDER.installed,
        }

    def gptq_audit_end(self) -> dict:
        for handle in getattr(self, "_gptq_audit_handles", []):
            handle.remove()
        self._gptq_audit_handles = []  # type: ignore[attr-defined]
        _RECORDER.enabled = False
        engine = _engine_snapshot(self)
        failures = _observed_failures()
        if not engine["enforce_eager"] or engine["cudagraph_mode"] != "NONE":
            failures.append(
                "observation engine is not eager "
                f"(enforce_eager={engine['enforce_eager']}, "
                f"cudagraph_mode={engine['cudagraph_mode']}); "
                "compiled or graphed replay would not re-enter the counters"
            )
        return {
            "selection_kind": "observed_call",
            "engine": engine,
            "forward_tags": _RECORDER.forward_tags,
            "calls": _RECORDER.snapshot(),
            "failures": failures,
            "instrumentation_enabled_during_recording_only": True,
        }

    def gptq_audit_uninstall(self) -> dict:
        uninstall_backend_counters()
        return {"installed": _RECORDER.installed}


def _gpu_worker():
    from vllm.v1.worker.gpu_worker import Worker

    return Worker


class GptqAuditWorker(_gpu_worker(), GptqAuditExtension):
    """Eager observation worker. Wrappers are installed before weight load."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        install_backend_counters()
        return super().load_model(load_dummy_weights=load_dummy_weights)
