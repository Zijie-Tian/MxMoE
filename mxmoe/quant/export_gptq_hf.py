"""Export MoE expert weights as a HuggingFace-style GPTQ checkpoint.

Phase-1 scope (uniform weight-only GPTQ):
  - bits in {1, 2, 3, 4, 8} for hf_float; pack format only {2, 3, 4, 8}
    (mixed / W-A / online-had → NotImplementedError)
  - Quantize MoE expert linears (gate/up/down + shared) only; keep attn/embed/norm/lm_head
  - Offline Hadamard (optional); online Had is rejected
  - Default write root: /mnt/raid/tzj/mxmoe/exported_hf_models/<model_name>/
    (override with env MXMOE_EXPORT_HF_ROOT or --save / --save-name)

Export formats:
  - hf_float (default): GPTQ fake-quant float weights + HF save_pretrained (always loadable)
  - gptq_pack: replace quantized Linear weights with qweight/qzeros/scales/g_idx in state_dict
               (AutoGPTQ/GPTQ-v1 layout; loading needs a GPTQ backend)
  - compressed_tensors: vLLM/Humming pack-quantized WNA16. Per-expert gate/up/down names
    stay unfused. Attention, embed, norm, lm_head, and the router stay dense.

Example (DeepSeek-V2-Lite, expert+dense W2A16 g128 asym, TP1 dense-FFN pad):
  python -m mxmoe.quant.export_gptq_hf \\
      --model ds2 --wbits 2 --groupsize 128 --export-format compressed_tensors \\
      --pad-dense-ffn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from project_config import CUR_DIR, ID2NAME
from mxmoe.quant.data_utils import get_wikitext2
from mxmoe.quant.gptq import GPTQ, Quantizer, find_layers
from mxmoe.quant.gptq_pack import (
    COMPRESSED_TENSORS_BITS,
    SUPPORTED_PACK_BITS,
    build_compressed_tensors_config,
    build_quantize_config_dict,
    compressed_tensors_linear_shapes,
    dense_ffn_alignment,
    dequant_weight,
    pad_quant_state_for_group_alignment,
    quant_state_to_compressed_tensors,
    quant_state_to_packed,
    validate_compressed_tensors_conversion,
)
from mxmoe.quant.moe_utils import load_hf_model, load_tokenizer

# Prefer RAID for large HF dumps (/mnt/data is often full). Override via env.
_DEFAULT_EXPORT_ROOT = Path("/mnt/raid/tzj/mxmoe/exported_hf_models")
EXPORT_HF_ROOT = Path(os.environ.get("MXMOE_EXPORT_HF_ROOT", str(_DEFAULT_EXPORT_ROOT)))

# Expert / MLP linear name filters (same spirit as gptq.py llama_sequential).
EXPERT_FILTERS = ["w1", "w2", "w3", "up_proj", "gate_proj", "down_proj"]


def _assert_exportable(args: argparse.Namespace) -> None:
    if args.wbits not in (1, 2, 3, 4, 8):
        raise NotImplementedError(
            f"Supported wbits for export: {{1,2,3,4,8}}, got {args.wbits}"
        )
    if args.export_format == "gptq_pack" and args.wbits not in SUPPORTED_PACK_BITS:
        raise NotImplementedError(
            f"gptq_pack only supports bits {SUPPORTED_PACK_BITS} (no standard W1 pack); "
            f"use --export-format hf_float for W{args.wbits}."
        )
    if args.export_format == "compressed_tensors":
        if args.wbits not in COMPRESSED_TENSORS_BITS:
            raise NotImplementedError(
                "compressed_tensors export supports bit widths "
                f"{COMPRESSED_TENSORS_BITS}; got {args.wbits}. "
                "W1/W3 are not packed as raw unsigned int32."
            )
        if args.hadamard or args.online_had:
            raise NotImplementedError(
                "compressed_tensors export is pure GPTQ; Hadamard/online rotation "
                "is not written into the checkpoint."
            )
        if args.act_order or args.no_static_groups:
            raise NotImplementedError(
                "compressed_tensors export requires static groups (desc_act=False). "
                "vLLM rejects group/dynamic act-order for this format."
            )
        if args.max_layers is not None:
            raise NotImplementedError(
                "compressed_tensors export writes a complete tree; "
                "--max-layers would omit layers vLLM still expects to be packed."
            )
    if args.online_had:
        raise NotImplementedError(
            "Online Hadamard cannot be packed into vanilla HF GPTQ; use --hadamard without --online-had"
        )
    if args.qconfig is not None:
        raise NotImplementedError(
            "Mixed-precision --qconfig export is not implemented yet (uniform Wk only)."
        )
    if args.quant_attn:
        raise NotImplementedError("--quant-attn export is deferred; experts only for now.")


def default_export_name(args: argparse.Namespace) -> str:
    sym = "sym" if args.sym else "asym"
    had = "-had" if args.hadamard else ""
    name = f"{args.model}-gptq{had}-w{args.wbits}-g{args.groupsize}-{sym}"
    if args.export_format == "compressed_tensors":
        name += "-ct"
        if getattr(args, "pad_dense_ffn", False):
            name += "-padffn"
    return name


def resolve_export_dir(args: argparse.Namespace) -> Path:
    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = Path(CUR_DIR) / p
        return p
    name = args.save_name or default_export_name(args)
    return EXPORT_HF_ROOT / name


@torch.no_grad()
def gptq_experts_sequential(
    model: nn.Module,
    dataloader: list[torch.Tensor],
    *,
    bits: int,
    groupsize: int,
    sym: bool,
    act_order: bool,
    static_groups: bool,
    percdamp: float,
    nsamples: int,
    dev: torch.device,
    max_layers: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Layer-wise GPTQ on expert/MLP linears. Returns name -> quant_state."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    ori_dev = model.model.embed_tokens.weight.device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    seqlen = dataloader[0].shape[1]
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(ori_dev)
    model.model.embed_tokens = model.model.embed_tokens.to(ori_dev)
    if rotary_emb is not None:
        model.model.rotary_emb = rotary_emb.to(ori_dev)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quant_states: dict[str, dict[str, torch.Tensor]] = {}
    n_layers = len(layers) if max_layers is None else min(len(layers), max_layers)

    for i in range(n_layers):
        if "cpu" in str(ori_dev):
            layer = layers[i].to(dev)
        else:
            layer = layers[i]

        full = find_layers(layer, filters=EXPERT_FILTERS)
        names = list(full.keys())
        print(f"[export] Layer {i}/{n_layers - 1}: quantizing {len(names)} linears (W{bits}A16 g{groupsize})")

        gptq_objs: dict[str, GPTQ] = {}
        for name in names:
            gptq_objs[name] = GPTQ(full[name])
            gptq_objs[name].quantizer = Quantizer()
            gptq_objs[name].quantizer.configure(bits, perchannel=True, sym=sym, mse=False)

        def make_hook(nm):
            def hook(_m, inp, out):
                gptq_objs[nm].add_batch(inp[0].data, out.data)
            return hook

        handles = [full[n].register_forward_hook(make_hook(n)) for n in names]
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        for h in handles:
            h.remove()

        for name in names:
            print(f"  GPTQ L{i}.{name} ...")
            st = gptq_objs[name].fasterquant(
                percdamp=percdamp,
                groupsize=groupsize,
                actorder=act_order,
                static_groups=static_groups,
                return_quant_state=True,
            )
            quant_states[f"model.layers.{i}.{name}"] = st
            gptq_objs[name].free()

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        if "cpu" in str(ori_dev):
            layers[i] = layer.to(ori_dev)
        else:
            layers[i] = layer
        del layer, gptq_objs
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quant_states


def _apply_offline_had(model: nn.Module, seed: int) -> None:
    """Offline QuaRot-style rotation (no online Had hooks)."""
    torch.manual_seed(seed)
    from mxmoe.quant.rotation import ModelRotator

    rotator = ModelRotator(model, "hadamard")
    rotator.rotate_model(model, enable_online_rotation=False)


def build_packed_state_dict(
    model: nn.Module,
    quant_states: dict[str, dict[str, torch.Tensor]],
    bits: int,
    scale_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Full state_dict with quantized linears replaced by GPTQ packed tensors."""
    sd = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    quantized_bases = set(quant_states.keys())

    for key, tensor in sd.items():
        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            if base in quantized_bases:
                continue
        if key.endswith(".bias"):
            base = key[: -len(".bias")]
            if base in quantized_bases:
                continue
        out[key] = tensor.detach().cpu()

    for base, st in quant_states.items():
        bias = None
        bias_key = f"{base}.bias"
        if bias_key in sd:
            bias = sd[bias_key]
        packed = quant_state_to_packed(st, bits, scale_dtype=scale_dtype, bias=bias)
        out[f"{base}.qweight"] = packed["qweight"]
        out[f"{base}.qzeros"] = packed["qzeros"]
        out[f"{base}.scales"] = packed["scales"]
        out[f"{base}.g_idx"] = packed["g_idx"]
        if "bias" in packed:
            out[f"{base}.bias"] = packed["bias"]
    return out


def verify_pack_roundtrip(
    quant_states: dict[str, dict[str, torch.Tensor]],
    bits: int,
    max_modules: int = 4,
) -> list[dict[str, Any]]:
    reports = []
    for i, (name, st) in enumerate(quant_states.items()):
        if i >= max_modules:
            break
        W = st["weight"]
        packed = quant_state_to_packed(st, bits, scale_dtype=torch.float16)
        W_hat = dequant_weight(
            packed["qweight"],
            packed["qzeros"],
            packed["scales"],
            packed["g_idx"],
            bits,
            out_features=W.shape[0],
            in_features=W.shape[1],
        )
        err = (W.float() - W_hat).abs()
        reports.append(
            {
                "name": name,
                "max_abs_err": float(err.max()),
                "mean_abs_err": float(err.mean()),
            }
        )
    return reports


def copy_trust_remote_code_files(src_model_path: str | Path, export_dir: Path) -> list[str]:
    """Copy custom modeling/tokenizer Python files required by trust_remote_code.

    ``save_pretrained`` often only ships config + weights; DeepSeek-V2 needs
    ``modeling_deepseek.py`` (and related) next to ``config.json``.
    """
    import shutil

    src = Path(src_model_path)
    export_dir = Path(export_dir)
    copied: list[str] = []
    if not src.is_dir():
        return copied
    for py in sorted(src.glob("*.py")):
        # Skip accidental caches
        if py.name.startswith("_"):
            continue
        dst = export_dir / py.name
        shutil.copy2(py, dst)
        copied.append(py.name)
    return copied

_DENSE_FFN_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$"
)


def dense_ffn_padding_kind(name: str, first_k: int) -> str | None:
    """Return how to pad one dense-MLP projection, or None if it is not one.

    Matches only ``layers.{i}.mlp.{gate,up,down}_proj`` with ``i < first_k``.
    Nested experts and merged gate_up names do not match. Gate and up are
    padded on their own output rows, not as one concatenated matrix.
    """
    match = _DENSE_FFN_RE.search(name)
    if match is None or int(match.group(1)) >= first_k:
        return None
    if match.group(2) in ("gate_proj", "up_proj"):
        return "output_rows"
    return "input_cols"


def require_compressed_tensors_alignment(
    quant_states: dict[str, dict[str, torch.Tensor]],
    *,
    bits: int,
    groupsize: int,
    symmetric: bool,
) -> None:
    """Reject a checkpoint vLLM would refuse, with a padding hint for layer 0."""
    errors = []
    for name, state in quant_states.items():
        out_features, in_features = (int(x) for x in state["weight"].shape)
        try:
            compressed_tensors_linear_shapes(
                out_features,
                in_features,
                bits,
                groupsize,
                symmetric=symmetric,
            )
        except ValueError as exc:
            hint = ""
            if (
                groupsize > 0
                and in_features % groupsize != 0
                and dense_ffn_padding_kind(name, 10**9) == "input_cols"
            ):
                hint = " Pass --pad-dense-ffn to fill this dense down_proj tail group."
            errors.append(f"{name}: {exc}.{hint}")
    if errors:
        raise ValueError(
            "compressed_tensors export refused an unaligned quant state:\n"
            + "\n".join(errors)
        )


def apply_dense_ffn_group_padding(
    model: nn.Module,
    quant_states: dict[str, dict[str, torch.Tensor]],
    *,
    bits: int,
    groupsize: int,
) -> dict[str, Any]:
    """Pad export-only quant states for the dense FFN. Does not touch the model."""
    original_intermediate = int(model.config.intermediate_size)
    original_moe = getattr(model.config, "moe_intermediate_size", None)
    first_k = int(getattr(model.config, "first_k_dense_replace", 0) or 0)
    plan = dense_ffn_alignment(
        original_intermediate, bits=bits, groupsize=groupsize
    )
    if original_intermediate == 10944 and groupsize == 128 and bits == 2:
        if plan["padded_intermediate_size"] != 11008 or plan["pad"] != 64:
            raise RuntimeError(f"ds2 W2 g128 dense-FFN padding contract failed: {plan}")
    named = dict(model.named_modules())
    modules: list[dict[str, Any]] = []
    for name in list(quant_states):
        kind = dense_ffn_padding_kind(name, first_k)
        if kind is None:
            continue
        module = named.get(name)
        if module is None or not hasattr(module, "weight"):
            raise KeyError(f"quantized dense FFN {name} is not a model module")
        before_shape = tuple(module.weight.shape)
        state = quant_states[name]
        out_features, in_features = (int(x) for x in state["weight"].shape)
        groups_before = int(state["scale"].shape[1])
        if kind == "output_rows" and out_features != original_intermediate:
            raise ValueError(
                f"{name} N={out_features} != intermediate_size {original_intermediate}"
            )
        if kind == "input_cols" and in_features != original_intermediate:
            raise ValueError(
                f"{name} K={in_features} != intermediate_size {original_intermediate}"
            )
        if plan["pad"]:
            quant_states[name] = pad_quant_state_for_group_alignment(
                state,
                kind=kind,
                pad=plan["pad"],
                bits=bits,
                groupsize=groupsize,
            )
        after = quant_states[name]["weight"]
        if tuple(module.weight.shape) != before_shape:
            raise RuntimeError(f"export padding mutated source weight {name}")
        modules.append(
            {
                "name": name,
                "kind": kind,
                "pad": plan["pad"],
                "shape_before": [out_features, in_features],
                "shape_after": [int(after.shape[0]), int(after.shape[1])],
                "scale_groups_before": groups_before,
                "scale_groups_after": int(quant_states[name]["scale"].shape[1]),
            }
        )
    if int(model.config.intermediate_size) != original_intermediate:
        raise RuntimeError("export padding mutated model.config.intermediate_size")
    if getattr(model.config, "moe_intermediate_size", None) != original_moe:
        raise RuntimeError("export padding mutated model.config.moe_intermediate_size")
    if plan["pad"] > 0 and not modules:
        raise ValueError(
            "--pad-dense-ffn found no dense gate/up/down quant states in the "
            f"first {first_k} layers"
        )
    applied = bool(plan["pad"]) and bool(modules)
    return {
        **plan,
        "applied": applied,
        "exported_intermediate_size": (
            plan["padded_intermediate_size"] if applied else original_intermediate
        ),
        "moe_intermediate_size": original_moe,
        "first_k_dense_replace": first_k,
        "modules": modules,
        "source_config_unchanged": True,
        "source_weights_unchanged": True,
    }


def verify_compressed_tensors_roundtrip(
    quant_states: dict[str, dict[str, torch.Tensor]],
    bits: int,
    *,
    symmetric: bool,
    groupsize: int,
    prefer_names: list[str],
) -> list[dict[str, Any]]:
    ordered = [name for name in prefer_names if name in quant_states]
    ordered.extend(name for name in quant_states if name not in ordered)
    limit = max(4, len(prefer_names))
    reports = []
    for name in ordered[:limit]:
        report = validate_compressed_tensors_conversion(
            quant_states[name],
            bits,
            symmetric=symmetric,
            groupsize=groupsize,
        )
        reports.append({"name": name, **report})
    return reports


def _cpu_owned(tensor: torch.Tensor, seen_storage: set[int]) -> torch.Tensor:
    cpu = tensor.detach().cpu().contiguous()
    storage_ptr = cpu.untyped_storage().data_ptr()
    if storage_ptr in seen_storage:
        cpu = cpu.clone()
        storage_ptr = cpu.untyped_storage().data_ptr()
    seen_storage.add(storage_ptr)
    return cpu


def build_compressed_tensors_state_dict(
    model: nn.Module,
    quant_states: dict[str, dict[str, torch.Tensor]],
    bits: int,
    *,
    symmetric: bool,
    groupsize: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, list[int]]]]:
    """Dense tensors plus packed linears. Drops each quant state after packing.

    ``weight_scale`` stays float32. The model dtype is not applied here.
    """
    bases = set(quant_states)
    seen_storage: set[int] = set()
    out: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if key.endswith(".weight") and key[: -len(".weight")] in bases:
            continue
        out[key] = _cpu_owned(tensor, seen_storage)
    shapes: dict[str, dict[str, list[int]]] = {}
    for base in list(quant_states):
        state = quant_states.pop(base)
        out_features, in_features = (int(x) for x in state["weight"].shape)
        expected = compressed_tensors_linear_shapes(
            out_features,
            in_features,
            bits,
            groupsize,
            symmetric=symmetric,
        )
        packed = quant_state_to_compressed_tensors(
            state,
            bits,
            symmetric=symmetric,
            groupsize=groupsize,
        )
        del state
        if packed["weight_scale"].dtype != torch.float32:
            raise RuntimeError(f"{base}.weight_scale is not float32")
        for suffix, tensor in packed.items():
            if tuple(tensor.shape) != expected[suffix]:
                raise RuntimeError(
                    f"{base}.{suffix} shape {tuple(tensor.shape)} != {expected[suffix]}"
                )
            out[f"{base}.{suffix}"] = tensor
        shapes[base] = {
            name: list(shape) for name, shape in expected.items()
        }
        shapes[base]["weight_shape_value"] = [out_features, in_features]
    return out, shapes


def _publish_export_dir(staging: Path, export_dir: Path) -> None:
    if not (staging / "EXPORT_COMPLETE").is_file():
        raise RuntimeError(f"refusing to publish incomplete export {staging}")
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    if not export_dir.exists():
        os.replace(staging, export_dir)
        return
    previous = export_dir.with_name(export_dir.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    os.replace(export_dir, previous)
    try:
        os.replace(staging, export_dir)
    except Exception:
        os.replace(previous, export_dir)
        raise
    shutil.rmtree(previous)


def _exported_config_dict(
    model: nn.Module,
    *,
    intermediate_size: int | None,
    quantization_config: dict[str, Any],
) -> dict[str, Any]:
    live_intermediate = int(model.config.intermediate_size)
    live_moe = getattr(model.config, "moe_intermediate_size", None)
    cfg = json.loads(model.config.to_json_string())
    if int(model.config.intermediate_size) != live_intermediate:
        raise RuntimeError("reading config mutated intermediate_size")
    if getattr(model.config, "moe_intermediate_size", None) != live_moe:
        raise RuntimeError("reading config mutated moe_intermediate_size")
    if intermediate_size is not None:
        cfg["intermediate_size"] = int(intermediate_size)
    if live_moe is not None and "moe_intermediate_size" in cfg:
        if cfg["moe_intermediate_size"] != live_moe:
            raise RuntimeError("exported config changed moe_intermediate_size")
    cfg["quantization_config"] = quantization_config
    torch_dtype = cfg.get("torch_dtype")
    if torch_dtype is not None and not isinstance(torch_dtype, str):
        cfg["torch_dtype"] = str(torch_dtype).replace("torch.", "")
    return cfg


def save_compressed_tensors_export(
    model: nn.Module,
    tokenizer,
    export_dir: Path,
    *,
    args: argparse.Namespace,
    quant_states: dict[str, dict[str, torch.Tensor]],
    quantize_config: dict[str, Any],
    meta: dict[str, Any],
    pack_report: list[dict[str, Any]],
) -> None:
    """Write a complete tree to a sibling directory, then publish it."""
    export_dir = Path(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = export_dir.with_name(export_dir.name + ".incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        packed_sd, module_shapes = build_compressed_tensors_state_dict(
            model,
            quant_states,
            args.wbits,
            symmetric=bool(args.sym),
            groupsize=args.groupsize,
        )
        from safetensors.torch import save_file

        save_file(packed_sd, str(staging / "model.safetensors"))
        del packed_sd
        padding = meta.get("dense_ffn_padding") or {}
        exported_intermediate = padding.get("exported_intermediate_size")
        cfg = _exported_config_dict(
            model,
            intermediate_size=exported_intermediate,
            quantization_config=quantize_config,
        )
        with open(staging / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        with open(staging / "quantize_config.json", "w", encoding="utf-8") as f:
            json.dump(quantize_config, f, indent=2)
            f.write("\n")
        with open(staging / "quantized_modules.json", "w", encoding="utf-8") as f:
            json.dump(module_shapes, f, indent=2)
            f.write("\n")
        with open(staging / "mxmoe_export_meta.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "pack_roundtrip": pack_report, "module_count": len(module_shapes)}, f, indent=2)
            f.write("\n")
        tokenizer.save_pretrained(staging)
        src_path = meta.get("model_path") or ID2NAME.get(args.model)
        copied: list[str] = []
        if src_path:
            copied = copy_trust_remote_code_files(src_path, staging)
            generation = Path(src_path) / "generation_config.json"
            if generation.is_file():
                shutil.copy2(generation, staging / "generation_config.json")
                copied.append(generation.name)
        pad_line = "dense_ffn_padding=none\n"
        if padding.get("applied"):
            pad_line = (
                "dense_ffn_padding="
                f"{padding['intermediate_size']}->{padding['exported_intermediate_size']}"
                f" pad={padding['pad']} moe_intermediate_size={padding.get('moe_intermediate_size')}\n"
            )
        (staging / "EXPORT_FORMAT.txt").write_text(
            "export_format=compressed_tensors\n"
            "quant_method=compressed-tensors format=pack-quantized quantization_status=compressed\n"
            "Per-projection tensors: weight_packed, weight_scale, weight_zero_point "
            "(asymmetric), weight_shape.\n"
            "weight_packed is [N, K/(32/bits)] unsigned codes along K. "
            "weight_scale is float32, the GPTQ scale; vLLM casts it at load. "
            "weight_zero_point is [N/(32/bits), G] raw zeros along N, not GPTQ-v1 zero-1.\n"
            "Padded zeros are encoded as q=z. Gate and up are padded separately.\n"
            "Attention, embed, norm, lm_head, and the router stay dense and are listed in ignore.\n"
            f"{pad_line}"
            f"remote_code={copied}\n",
            encoding="utf-8",
        )
        (staging / "EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
        _publish_export_dir(staging, export_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[export] saved compressed-tensors → {export_dir}")



def save_export(
    model: nn.Module,
    tokenizer,
    export_dir: Path,
    *,
    args: argparse.Namespace,
    quant_states: dict[str, dict[str, torch.Tensor]],
    quantize_config: dict[str, Any],
    meta: dict[str, Any],
    pack_report: list[dict[str, Any]],
) -> None:
    if args.export_format == "compressed_tensors":
        save_compressed_tensors_export(
            model,
            tokenizer,
            export_dir,
            args=args,
            quant_states=quant_states,
            quantize_config=quantize_config,
            meta=meta,
            pack_report=pack_report,
        )
        return
    export_dir.mkdir(parents=True, exist_ok=True)

    with open(export_dir / "quantize_config.json", "w", encoding="utf-8") as f:
        json.dump(quantize_config, f, indent=2)
    with open(export_dir / "mxmoe_export_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "pack_roundtrip": pack_report}, f, indent=2)

    tokenizer.save_pretrained(export_dir)

    if args.export_format == "hf_float":
        model.config.torch_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        model.save_pretrained(export_dir, safe_serialization=True)
        (export_dir / "EXPORT_FORMAT.txt").write_text(
            "export_format=hf_float\n"
            "Weights are GPTQ-optimized fake-quant floats (standard nn.Linear).\n"
            "See quantize_config.json and mxmoe_export_meta.json for GPTQ hyper-parameters.\n"
            "For packed qweight tensors use --export-format gptq_pack.\n",
            encoding="utf-8",
        )
    elif args.export_format == "gptq_pack":
        scale_dtype = torch.float16
        packed_sd = build_packed_state_dict(model, quant_states, args.wbits, scale_dtype)
        model.config.quantization_config = quantize_config
        model.config.save_pretrained(export_dir)
        try:
            from safetensors.torch import save_file

            save_file(packed_sd, str(export_dir / "model.safetensors"))
        except Exception:
            torch.save(packed_sd, export_dir / "pytorch_model.bin")
        (export_dir / "EXPORT_FORMAT.txt").write_text(
            "export_format=gptq_pack\n"
            "Linear expert weights stored as qweight/qzeros/scales/g_idx (GPTQ v1 layout).\n"
            "Loading requires a GPTQ backend (e.g. gptqmodel). Non-expert tensors remain dense.\n",
            encoding="utf-8",
        )
        with open(export_dir / "quantized_modules.json", "w", encoding="utf-8") as f:
            json.dump(sorted(quant_states.keys()), f, indent=2)
    else:
        raise NotImplementedError(args.export_format)

    # Always copy remote-code .py from the source checkpoint (DeepSeek etc.)
    src_path = meta.get("model_path") or ID2NAME.get(args.model)
    if src_path:
        copied = copy_trust_remote_code_files(src_path, export_dir)
        print(f"[export] copied remote-code files: {copied}")

    print(f"[export] saved → {export_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    known = ["ds2", "qwen2_moe", "mixtral", "qwen2_moe_57b"]
    try:
        known = sorted(set(list(ID2NAME.keys()) + known))
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Export uniform GPTQ MoE experts to HF directory")
    p.add_argument("--model", type=str, default="ds2", choices=known)
    p.add_argument("--wbits", type=int, default=2, choices=[1, 2, 3, 4, 8])
    p.add_argument("--groupsize", type=int, default=128)
    p.add_argument("--sym", action="store_true", help="Symmetric weight quant (default: asym)")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--act-order", action="store_true", help="GPTQ act-order (desc_act)")
    p.add_argument(
        "--no-static-groups",
        action="store_true",
        help="Disable static groups (default: static groups ON when groupsize>0)",
    )
    p.add_argument("--hadamard", action="store_true", help="Offline Hadamard (QuaRot-style)")
    p.add_argument("--online-had", action="store_true", help="Rejected for HF export")
    p.add_argument("--qconfig", type=str, default=None, help="Mixed qconfig (NotImplemented)")
    p.add_argument("--quant-attn", action="store_true", help="NotImplemented in phase-1")
    p.add_argument(
        "--export-format",
        type=str,
        default="hf_float",
        choices=["hf_float", "gptq_pack", "compressed_tensors"],
    )
    p.add_argument(
        "--pad-dense-ffn",
        action="store_true",
        help=(
            "Pad the dense MLP intermediate size up to a multiple of the group "
            "size and pack factor. For ds2 W2 g128 this writes intermediate_size "
            "11008: gate/up gain 64 output rows and down gains 64 input columns. "
            "Does not change moe_intermediate_size or the source model."
        ),
    )
    p.add_argument("--save", type=str, default=None, help="Full export path override")
    p.add_argument("--save-name", type=str, default=None, help="Subdir under EXPORT_HF_ROOT")
    p.add_argument("--max-layers", type=int, default=None, help="Only quantize first N layers (smoke)")
    p.add_argument("--device-map", type=str, default=None, help="Force device_map, e.g. cpu")
    p.add_argument(
        "--eval-ppl",
        action="store_true",
        help="After export, load the saved HF model and run wikitext2 PPL",
    )
    p.add_argument("--ppl-input-len", type=int, default=4096)
    return p.parse_args(argv)


def eval_exported_ppl(export_dir: Path, model_id: str, input_len: int = 4096) -> float:
    """Load exported hf_float checkpoint and report wikitext2 PPL."""
    from mxmoe.quant.evaluator import Evaluator

    print(f"[export] eval PPL from {export_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        export_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    evaluator = Evaluator(tokenizer, model_id)
    ppl = evaluator.eval_ppl(model, input_len=input_len)
    print(f"[export] PPL={ppl:.4f}")
    return ppl


def main(argv=None) -> None:
    args = parse_args(argv)
    _assert_exportable(args)
    transformers.set_seed(args.seed)

    export_dir = resolve_export_dir(args)
    print(
        f"[export] model={args.model} W{args.wbits}A16 g{args.groupsize} "
        f"sym={args.sym} had={args.hadamard} format={args.export_format}"
    )
    print(f"[export] root={EXPORT_HF_ROOT}")
    print(f"[export] → {export_dir}")

    device_map = (
        "cpu"
        if args.model in ["qwen2_moe", "qwen2_moe_57b", "mixtral"] or args.device_map == "cpu"
        else None
    )
    model = load_hf_model(args.model, device_map=device_map)

    tokenizer = load_tokenizer(args.model)
    trainloader, _ = get_wikitext2(
        args.nsamples, args.seed, args.seqlen, tokenizer, args.model, test_only=False
    )

    if args.hadamard:
        print("[export] applying offline Hadamard rotation ...")
        t0 = time.time()
        _apply_offline_had(model, args.seed)
        print(f"[export] hadamard done in {time.time() - t0:.1f}s")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    act_order = bool(args.act_order)
    static_groups = (not args.no_static_groups) or act_order

    t0 = time.time()
    quant_states = gptq_experts_sequential(
        model,
        trainloader,
        bits=args.wbits,
        groupsize=args.groupsize,
        sym=args.sym,
        act_order=act_order,
        static_groups=static_groups,
        percdamp=args.percdamp,
        nsamples=args.nsamples,
        dev=dev,
        max_layers=args.max_layers,
    )
    print(f"[export] GPTQ finished in {time.time() - t0:.1f}s; modules={len(quant_states)}")

    padding_report = None
    scale_dtype = torch.float16
    if args.export_format == "compressed_tensors":
        # Keep GPTQ FP32 scales on disk. Do not narrow them to the model dtype.
        scale_dtype = torch.float32
        if args.pad_dense_ffn:
            padding_report = apply_dense_ffn_group_padding(
                model,
                quant_states,
                bits=args.wbits,
                groupsize=args.groupsize,
            )
            print(f"[export] dense FFN padding: {padding_report}")
        require_compressed_tensors_alignment(
            quant_states,
            bits=args.wbits,
            groupsize=args.groupsize,
            symmetric=bool(args.sym),
        )
        prefer = [item["name"] for item in (padding_report or {}).get("modules", [])]
        pack_report = verify_compressed_tensors_roundtrip(
            quant_states,
            args.wbits,
            symmetric=bool(args.sym),
            groupsize=args.groupsize,
            prefer_names=prefer,
        )
        print(f"[export] compressed-tensors round-trip (sample): {pack_report}")
        quantize_config = build_compressed_tensors_config(
            bits=args.wbits,
            group_size=args.groupsize,
            sym=bool(args.sym),
        )
    elif args.wbits in SUPPORTED_PACK_BITS:
        pack_report = verify_pack_roundtrip(quant_states, args.wbits, max_modules=4)
        print(f"[export] pack round-trip (sample): {pack_report}")
        quantize_config = build_quantize_config_dict(
            bits=args.wbits,
            group_size=args.groupsize,
            sym=args.sym,
            desc_act=act_order,
            damp_percent=args.percdamp,
            meta={
                "mxmoe_export_format": args.export_format,
                "hadamard": args.hadamard,
                "online_had": False,
                "quantized_scope": "moe_experts_only",
            },
        )
    else:
        pack_report = [{"note": f"pack skipped for wbits={args.wbits}"}]
        print(f"[export] pack round-trip skipped (wbits={args.wbits})")
        quantize_config = build_quantize_config_dict(
            bits=args.wbits,
            group_size=args.groupsize,
            sym=args.sym,
            desc_act=act_order,
            damp_percent=args.percdamp,
            meta={
                "mxmoe_export_format": args.export_format,
                "hadamard": args.hadamard,
                "online_had": False,
                "quantized_scope": "moe_experts_only",
            },
        )
    meta = {
        "model_id": args.model,
        "model_path": ID2NAME[args.model],
        "wbits": args.wbits,
        "abits": 16,
        "groupsize": args.groupsize,
        "sym": args.sym,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "hadamard": args.hadamard,
        "online_had": False,
        "act_order": act_order,
        "export_format": args.export_format,
        "num_quantized_modules": len(quant_states),
        "max_layers": args.max_layers,
        "export_dir": str(export_dir),
        "export_root": str(EXPORT_HF_ROOT),
        "scale_dtype": str(scale_dtype).replace("torch.", ""),
        "dense_ffn_padding": padding_report,
        "zero_point_encoding": (
            "raw_gptq_additive_zero"
            if args.export_format == "compressed_tensors" and not args.sym
            else None
        ),
    }

    save_export(
        model,
        tokenizer,
        export_dir,
        args=args,
        quant_states=quant_states,
        quantize_config=quantize_config,
        meta=meta,
        pack_report=pack_report,
    )

    if args.eval_ppl:
        if args.export_format != "hf_float":
            raise NotImplementedError("--eval-ppl currently requires --export-format hf_float")
        # Free GPU memory before reload
        del model
        torch.cuda.empty_cache()
        try:
            ppl = eval_exported_ppl(export_dir, args.model, input_len=args.ppl_input_len)
            with open(export_dir / "eval_ppl.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"ppl": ppl, "input_len": args.ppl_input_len, "dataset": "wikitext2"},
                    f,
                    indent=2,
                )
            print(f"[export] wrote {export_dir / 'eval_ppl.json'}")
        except Exception as e:
            # Do not fail the whole multi-bit batch if only eval fails after a good save.
            print(f"[export] WARN: eval_ppl failed after successful save: {e}")
            with open(export_dir / "eval_ppl_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()
