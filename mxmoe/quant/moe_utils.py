import torch
import torch.nn as nn
from transformers import PreTrainedModel

class _LinearExpert(nn.Module):
    """Expose packed HF weights as real linears for GPTQ and activation hooks."""

    def __init__(self, packed, expert_idx, names):
        super().__init__()
        self.projection_names = names
        self.act_fn = packed.act_fn
        fused = packed.gate_up_proj[expert_idx].detach()
        midpoint = fused.shape[0] // 2
        projections = (
            fused[:midpoint].clone(),
            fused[midpoint:].clone(),
            packed.down_proj[expert_idx].detach().clone(),
        )
        for name, weight in zip(names, projections):
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
            linear.weight = nn.Parameter(weight, requires_grad=False)
            self.add_module(name, linear)

    def forward(self, hidden_states):
        gate, up, down = (getattr(self, name) for name in self.projection_names)
        return down(self.act_fn(gate(hidden_states)) * up(hidden_states))


class _LinearExperts(nn.ModuleList):
    """Preserve HF routing while each expert projection executes its own hooks."""

    def __init__(self, packed, names):
        experts = [_LinearExpert(packed, i, names) for i in range(packed.num_experts)]
        del packed.gate_up_proj, packed.down_proj
        super().__init__(experts)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=len(self) + 1).permute(2, 1, 0)
            active = mask.sum(dim=(-1, -2)).nonzero().flatten()
        for expert_idx in active:
            if expert_idx == len(self):
                continue
            slot, token = torch.where(mask[expert_idx])
            output = self[int(expert_idx)](hidden_states[token])
            output = output * top_k_weights[token, slot, None]
            result.index_add_(0, token, output.to(result.dtype))
        return result


@torch.no_grad()
def move_weight_to_cpu(model: PreTrainedModel, layer_idx: int = -1) -> list[dict[str, torch.Tensor]]:
    weights = [{} for _ in range(len(model.model.layers))]

    for layer_i, layer in enumerate(model.model.layers):
        for p_name, p in layer.named_parameters():
            if layer_idx != -1 and layer_idx != layer_i: continue
            weights[layer_i][p_name] = p.to("cpu")

    return weights

@torch.no_grad()
def recover_weight_from_cpu(model: PreTrainedModel, weights: list[list[torch.Tensor]], layer_idx: int = -1, exp_idx: int=-1):
    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue
        for p_name, p in layer.named_parameters():
            p.data.copy_(weights[layer_i][p_name])


def is_non_moe_layer(model_id: str, layer_idx: int):
    if model_id == "mixtral":
        return False
    elif model_id in ["qwen2_moe", "qwen2_moe_57b"]:
        return False
    elif model_id == "ds2":
        # TODO:
        return layer_idx == 0
    else:
        raise ValueError(f"Unsupported model: {model_id}")


@torch.no_grad()
def get_linear_block_weight(model_id:str, model_weight: dict|PreTrainedModel, layer_idx: int, expert_idx: int, linear_block_name: str, return_key=False) -> torch.Tensor:
    '''
    linear_block_name: [gate, up, down]
    '''
    if isinstance(model_weight, PreTrainedModel):
        model_weight = dict(model_weight.named_parameters())

    mlp_block_name = MOE_MLP_NAME_MAP[model_id]
    linear_block_name = MOE_WEIGHT_NAME_MAP[model_id][linear_block_name]

    if is_non_moe_layer(model_id, layer_idx):
        mlp_weight_key = f"{mlp_block_name}.{linear_block_name}"
    else:
        mlp_weight_key = f"{mlp_block_name}.experts.{expert_idx}.{linear_block_name}"

        if model_id == "mixtral" and f"model.layers.{layer_idx}.{mlp_weight_key}.weight" not in model_weight:
            # Older prequantized dictionaries retain the original checkpoint prefix.
            original_key = mlp_weight_key.replace("mlp.", "block_sparse_moe.", 1)
            if f"model.layers.{layer_idx}.{original_key}.weight" in model_weight:
                mlp_weight_key = original_key
        # handle the shared expert
        if model_weight.get(f"model.layers.{layer_idx}.{mlp_weight_key}.weight", None) is None:
            if model_id == "ds2":
                mlp_weight_key = f"{mlp_block_name}.shared_experts.{linear_block_name}"
            elif model_id in ["qwen2_moe", "qwen2_moe_57b"]:
                mlp_weight_key = f"{mlp_block_name}.shared_expert.{linear_block_name}"
            else:
                raise ValueError(f"Unsupported model: {model_id}")
    weight_key = f"model.layers.{layer_idx}.{mlp_weight_key}.weight"

    if return_key: return weight_key, model_weight[weight_key]
    return model_weight[weight_key]


@torch.no_grad()
def offload_moe_weights(model_id:str, model: PreTrainedModel, layer_id:int=-1, expert_id:int=-1, linear_block:str=""):
    cpu_weights: dict[str, torch.Tensor] = {}
    for layer_idx in range(len(model.model.layers)):
        if layer_id != -1 and layer_id != layer_idx: continue
        for expert_idx in range(len(get_expert_linears(model, layer_idx))):
            if expert_id != -1 and expert_id != expert_idx: continue
            for linear_block_name in ["gate", "up", "down"]:
                if len(linear_block) > 0 and linear_block_name != linear_block: continue
                wname,w = get_linear_block_weight(model_id, model, layer_idx, expert_idx, linear_block_name, True)
                cpu_weights[wname] = w.cpu()

    return cpu_weights

@torch.no_grad()
def substitue_moe_weights(model_id:str, model: PreTrainedModel, cpu_weights: dict[str, torch.Tensor], layer_id:int=-1, expert_id:int=-1, linear_block:str=""):
    for layer_idx in range(len(model.model.layers)):
        if layer_id != -1 and layer_id != layer_idx: continue
        for expert_idx in range(len(get_expert_linears(model, layer_idx))):
            if expert_id != -1 and expert_id != expert_idx: continue
            for linear_block_name in ["gate", "up", "down"]:
                if len(linear_block) > 0 and linear_block_name != linear_block: continue

                old_weight = get_linear_block_weight(model_id, model, layer_idx, expert_idx, linear_block_name)
                linear_block_weight = get_linear_block_weight(model_id, cpu_weights, layer_idx, expert_idx, linear_block_name)
                old_weight.copy_(linear_block_weight)


def get_expert_linears(model: PreTrainedModel, layer_idx: int = -1, exclude_non_moe_layer: bool=False) -> list[list[nn.Module]] | list[nn.Module]:
    '''
    return: the list of experts in MoE module of each layer

    exclude_non_moe_layer: for deepseek_v2, the first layer is traditional MLP layer
    '''
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    experts = [[] for _ in range(len(model.model.layers))]
    model_type = model.config.model_type

    if model_type not in ("deepseek_v2", "mixtral", "qwen2_moe"):
        raise NotImplementedError(f"Unsupported model type: {model_type}")
    names = ("w1", "w3", "w2") if model_type == "mixtral" else ("gate_proj", "up_proj", "down_proj")
    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i:
            continue
        block = layer.mlp
        if not hasattr(block, "experts"):
            if not exclude_non_moe_layer:
                experts[layer_i].append(block)
            continue
        if hasattr(block.experts, "gate_up_proj"):
            block.experts = _LinearExperts(block.experts, names)
        experts[layer_i].extend(block.experts)
        if model_type == "deepseek_v2":
            experts[layer_i].append(block.shared_experts)
        elif model_type == "qwen2_moe":
            experts[layer_i].append(block.shared_expert)
    if layer_idx != -1:
        experts = experts[layer_idx]
    return experts


def get_moe_gate_linears(model: PreTrainedModel, layer_idx: int = -1) -> list[nn.Module] | nn.Module:
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    expert_gates = [[] for _ in range(len(model.model.layers))]
    model_type = model.config.model_type

    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue
        if model_type not in ("deepseek_v2", "mixtral", "qwen2_moe"):
            raise NotImplementedError(f"Unsupported model type: {model_type}")
        if not hasattr(layer.mlp, "experts"):
            continue
        # Transformers 5 routers are not nn.Linear; rotate their projection parameter.
        expert_gates[layer_i].append(layer.mlp.gate.weight)
        if model_type == "qwen2_moe":
            expert_gates[layer_i].append(layer.mlp.shared_expert_gate)
    if layer_idx != -1:
        expert_gates = expert_gates[layer_idx]
    return expert_gates


def get_attn_linears(model: PreTrainedModel, layer_idx: int = -1) -> list[dict[str, nn.Module]] | dict[str, nn.Module]:
    '''
    return: the list of weights in attention module of each layer

    '''
    assert layer_idx < len(model.model.layers) or layer_idx == -1, f"layer_idx {layer_idx} is out of range"

    weights = [{} for _ in range(len(model.model.layers))]

    for layer_i, layer in enumerate(model.model.layers):
        if layer_idx != -1 and layer_idx != layer_i: continue

        for m_name, m in layer.self_attn.named_modules():
            if not isinstance(m, nn.Linear): continue
            weights[layer_i][m_name] = m

    if layer_idx != -1:
        weights = weights[layer_idx]
    return weights

MOE_WEIGHT_NAME_REMAP = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "w1": "gate",
    "w3": "up",
    "w2": "down",
}

MOE_MLP_NAME_MAP = {
    "ds2": "mlp",
    "qwen2_moe": "mlp",
    "qwen2_moe_57b": "mlp",
    "mixtral": "mlp",
}
MOE_WEIGHT_NAME_MAP = {
    **dict.fromkeys(["ds2", "qwen2_moe", "qwen2_moe_57b"], {
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }),
    "mixtral": {
        "gate": "w1",
        "up": "w3",
        "down": "w2",
    }
}

def get_linears_in_one_expert(expert: nn.Module) -> dict[str, nn.Linear]:
    '''
    return: the list of weights in one expert
    '''
    filters = [
        "gate_proj", "up_proj", "down_proj",
        "w1", "w2", "w3", # for mixtral
    ]

    weights = {}
    for m_name, m in expert.named_modules():
        if any(k in m_name for k in filters): 
            wname = MOE_WEIGHT_NAME_REMAP[m_name]
            weights[wname] = m

    return weights


def get_device_map(model_id: str):
    """Multi-GPU hand maps for known large models; otherwise ``\"auto\"``."""
    device_map = "auto"
    if torch.cuda.device_count() == 2:
        if model_id == "qwen2_moe_57b":
            device_map = {
                "model.embed_tokens": "cuda:0",
                "model.rotary_emb": "cuda:0",
                **{
                    f"model.layers.{k}": 0 for k in range(0, 14)
                },
                **{
                    f"model.layers.{k}": 1 for k in range(14, 28)
                },
                "model.norm": "cuda:1",
                "lm_head": 1,
            }
        elif model_id == "mixtral":
            device_map = {
                "model.embed_tokens": "cuda:0",
                **{
                    f"model.layers.{k}": 0 for k in range(0, 16)
                },
                **{
                    f"model.layers.{k}": 1 for k in range(16, 32)
                },
                "model.norm": "cuda:1",
                "lm_head": 1,
            }
    elif torch.cuda.device_count() == 4:
        if model_id == "qwen2_moe_57b":
            device_map = {
                "model.embed_tokens": "cuda:0",
                "model.rotary_emb": "cuda:0",
                **{f"model.layers.{k}": 0 for k in range(0, 7)},
                **{f"model.layers.{k}": 1 for k in range(7, 14)},
                **{f"model.layers.{k}": 2 for k in range(14, 21)},
                **{f"model.layers.{k}": 3 for k in range(21, 28)},
                "model.norm": "cuda:3",
                "lm_head": 3,
            }
        elif model_id == "mixtral":
            device_map = {
                "model.embed_tokens": "cuda:0",
                **{f"model.layers.{k}": 0 for k in range(0, 8)},
                **{f"model.layers.{k}": 1 for k in range(8, 16)},
                **{f"model.layers.{k}": 2 for k in range(16, 24)},
                **{f"model.layers.{k}": 3 for k in range(24, 32)},
                "model.norm": "cuda:3",
                "lm_head": 3,
            }

    return device_map


def build_max_memory(
    gpu_limit: str | None = None,
    cpu_limit: str | None = None,
    gpu_id: int = 0,
) -> dict[int | str, str] | None:
    """Build ``max_memory`` for accelerate ``load_checkpoint_and_dispatch``.

    Used for single-GPU layer placement with CPU (and optional disk) spill when
    the full model does not fit in VRAM. Placement is performed by accelerate
    via ``device_map=\"auto\"`` + this budget — not by hand-written layer moves.

    Args:
        gpu_limit: e.g. ``\"16GiB\"`` for GPU ``gpu_id``. Keep enough VRAM free for
            activations — 4k-token PPL logits on a 151k vocab need ~5GiB, so budget
            ~16GiB on a 24GB RTX 3090 rather than 20GiB+.
        cpu_limit: e.g. ``\"64GiB\"``. If only ``gpu_limit`` is set, defaults to ``\"128GiB\"``.
        gpu_id: physical/logical GPU index key expected by accelerate (usually 0).

    Returns:
        ``None`` if neither limit is provided (caller uses default device_map only).
    """
    if gpu_limit is None and cpu_limit is None:
        return None

    max_memory: dict[int | str, str] = {}
    if gpu_limit is not None:
        max_memory[gpu_id] = gpu_limit
    if cpu_limit is not None:
        max_memory["cpu"] = cpu_limit
    elif gpu_limit is not None:
        # Allow remaining modules to spill to host RAM.
        max_memory["cpu"] = "128GiB"
    return max_memory


def resolve_device_map(
    model_id: str,
    device_map: str | dict | None = None,
    max_memory: dict | None = None,
):
    """Choose device_map for accelerate dispatch.

    When ``max_memory`` is set (single-GPU fit), force ``\"auto\"`` so accelerate
    places whole modules under the budget. Otherwise keep multi-GPU hand maps.
    """
    if device_map is not None:
        return device_map
    if max_memory is not None:
        return "auto"
    return get_device_map(model_id)


def _normalize_legacy_tied_keys(module: nn.Module) -> None:
    """Transformers 5 expects ``_tied_weights_keys`` to be a dict, not a list."""
    tied = getattr(module, "_tied_weights_keys", None)
    if isinstance(tied, list):
        module._tied_weights_keys = {key: key for key in tied}
    for child in module.children():
        _normalize_legacy_tied_keys(child)


def _install_remote_code_shims() -> None:
    """DeepSeek-V2-Lite remote code imports symbols removed in Transformers 5."""
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False


def load_hf_model(
    model_id: str,
    ckpt=None,
    dtype="auto",
    *,
    device_map: str | dict | None = None,
    max_memory: dict | None = None,
    offload_folder: str | None = None,
    offload_buffers: bool = False,
):
    """Load native HF models, expose expert linears, then dispatch with accelerate.

    HF converts original or packed checkpoints on CPU before expert unpacking.
    Offload hooks are installed only after the module layout is final. Host RAM
    must fit the float weights; quantization uses its separate per-layer GPU path.
    """
    from project_config import ID2NAME
    from transformers import AutoModelForCausalLM, AutoConfig
    from accelerate import dispatch_model, infer_auto_device_map

    remote = model_id == "ds2"
    if remote:
        _install_remote_code_shims()
    model_name = ckpt if ckpt is not None else ID2NAME[model_id]
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=remote)
    # Native Qwen/Mixtral checkpoints may still advertise obsolete remote code.
    # DeepSeek-V2-Lite must retain its auto_map and modeling_deepseek.py.
    if not remote:
        config.__dict__.pop("auto_map", None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        # Remote DeepSeek-V2 does not declare `_supports_flash_attn`, so
        # Transformers 5 rejects FA2. Its own attention implementation still runs.
        attn_implementation="eager" if remote else "flash_attention_2",
        trust_remote_code=remote,
        device_map="cpu",
    )
    if remote:
        _normalize_legacy_tied_keys(model)
    get_expert_linears(model)

    resolved_map = resolve_device_map(model_id, device_map=device_map, max_memory=max_memory)
    if isinstance(resolved_map, str):
        if resolved_map in ("auto", "balanced", "balanced_low_0", "sequential"):
            from accelerate.utils import get_balanced_memory

            no_split = model._no_split_modules
            if resolved_map != "sequential":
                max_memory = get_balanced_memory(
                    model, max_memory=max_memory, no_split_module_classes=no_split,
                    dtype=next(model.parameters()).dtype,
                    low_zero=resolved_map == "balanced_low_0",
                )
            resolved_map = infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=no_split,
                dtype=next(model.parameters()).dtype, offload_buffers=offload_buffers,
            )
        else:
            resolved_map = {"": resolved_map}
    model = dispatch_model(
        model, device_map=resolved_map, offload_dir=offload_folder,
        offload_buffers=offload_buffers,
    )

    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is not None:
        # Summarize placement without dumping every key twice.
        from collections import Counter
        counts = Counter(str(v) for v in hf_map.values())
        print(f"[load_hf_model] hf_device_map device counts: {dict(counts)}")
        print(f"[load_hf_model] hf_device_map: {hf_map}")

    return model


def load_tokenizer(model_id: str):
    from project_config import ID2NAME
    from transformers import AutoTokenizer

    model_name = ID2NAME[model_id]
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)