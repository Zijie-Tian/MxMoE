"""Immutable routed-expert activation plans from the existing MxMoE qconfig schema.

Weights are not requantized: the selected entries must describe the CT checkpoint
actually being loaded. Shared/dense entries are not consumed by this adapter.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from mxmoe.kernels.qconfig import QLinearConfig

PROJECTIONS = ("gate", "up", "down")
_MODE_MASK = {4: 1, 8: 2, 16: 4}
_REQUIRED = frozenset(("w_bits", "w_gsize", "w_sym", "a_bits", "a_gsize", "a_sym", "a_clip"))


@dataclass(frozen=True)
class ActivationScheme:
    bits: int
    clip: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class ProjectionActivationPlan:
    schemes: tuple[ActivationScheme, ...]
    mask: int
    packed_bits: int

    @classmethod
    def from_schemes(cls, schemes):
        schemes = tuple(schemes)
        bits = {scheme.bits for scheme in schemes}
        return cls(schemes, sum(_MODE_MASK[value] for value in bits), max(bits - {16}, default=0))

    @property
    def quantized(self) -> bool:
        return bool(self.mask & 3)


@dataclass(frozen=True)
class LayerActivationPlan:
    layer_index: int
    gate: ProjectionActivationPlan
    up: ProjectionActivationPlan
    down: ProjectionActivationPlan

    @property
    def projections(self):
        return self.gate, self.up, self.down

    @property
    def gate_up_shared(self) -> bool:
        return self.gate == self.up

    @property
    def quantized(self) -> bool:
        return any(projection.quantized for projection in self.projections)

    def to_device(self, device) -> DeviceActivationPlan:
        bits = torch.tensor(
            [[scheme.bits for scheme in projection.schemes] for projection in self.projections],
            dtype=torch.int32, device=device,
        )
        clips = torch.tensor(
            [[scheme.clip for scheme in projection.schemes] for projection in self.projections],
            dtype=torch.float32, device=device,
        )
        return DeviceActivationPlan(self, bits, clips)


@dataclass(frozen=True)
class DeviceActivationPlan:
    plan: LayerActivationPlan
    bits: torch.Tensor  # [gate/up/down, E], each projection contiguous
    clips: torch.Tensor  # [gate/up/down, E, negative/positive]


@lru_cache(maxsize=8)
def _read_config(path: str, mtime_ns: int, size: int):
    # Keyed by file identity, not just its name: a later engine must not inherit
    # the previous engine's plan after the operator changes a config in place.
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("MxMoE activation config must be a layer-indexed JSON object")
    return result


def load_activation_plan(
    config_path: str, layer_name: str, num_experts: int, weight_bits: int,
    weight_group_size: int = 128, weight_symmetric: bool = False,
) -> LayerActivationPlan:
    """Read every routed gate/up/down explicitly; never infer A from W or default A16."""
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name)
    if match is None:
        raise ValueError(f"Cannot resolve MxMoE config layer index from {layer_name!r}")
    if num_experts <= 0:
        raise ValueError("MxMoE activation plan requires at least one routed expert")
    path = Path(config_path).expanduser().resolve()
    stat = path.stat()
    source = _read_config(str(path), stat.st_mtime_ns, stat.st_size)
    layer_index = int(match.group(1))
    try:
        experts = source[str(layer_index)]["experts"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"MxMoE activation config is missing layer {layer_index}") from exc
    projections = []
    for name in PROJECTIONS:
        schemes = []
        for expert in range(num_experts):
            where = f"layer {layer_index}, expert {expert}, {name}"
            try:
                fields = experts[str(expert)][name]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"MxMoE activation config is missing {where}") from exc
            if not isinstance(fields, dict) or not _REQUIRED.issubset(fields):
                missing = sorted(_REQUIRED - fields.keys()) if isinstance(fields, dict) else sorted(_REQUIRED)
                raise ValueError(f"{where}: missing explicit qconfig fields {missing}")
            cfg = QLinearConfig(**fields)
            actual = (weight_bits, weight_group_size, weight_symmetric)
            requested = (cfg.w_bits, cfg.w_gsize, cfg.w_sym)
            if requested != actual:
                raise ValueError(
                    f"{where}: weight config {requested} does not match checkpoint {actual}; "
                    "activation configuration cannot requantize or change checkpoint weights"
                )
            if type(cfg.a_bits) is not int or cfg.a_bits not in _MODE_MASK:
                raise ValueError(f"{where}: activation precision must be A4, A8 or A16, got {cfg.a_bits}")
            if weight_bits == 8 and cfg.a_bits == 4:
                raise ValueError(f"{where}: W8A4 is unsupported; use A8 or A16 for W8 weights")
            if cfg.a_bits == 16:
                schemes.append(ActivationScheme(16))
                continue
            if cfg.a_sym is not True or cfg.a_gsize != -1:
                raise ValueError(f"{where}: A4/A8 requires symmetric per-token activations (a_sym=true, a_gsize=-1)")
            clip = cfg.a_clip
            if (not isinstance(clip, (tuple, list)) or len(clip) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 0 < value <= 1 for value in clip)):
                raise ValueError(f"{where}: a_clip must contain two finite factors in (0, 1]")
            schemes.append(ActivationScheme(cfg.a_bits, tuple(float(value) for value in clip)))
        projections.append(ProjectionActivationPlan.from_schemes(schemes))
    return LayerActivationPlan(layer_index, *projections)
