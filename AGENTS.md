# MxMoE project agent notes

## Documentation policy (`docs/`)

- **Do not** use `docs/` (or other in-repo markdown) to record environment setup, package installs, conda/CUDA configuration, machine-local paths, or similar ops/runbooks.
- Those notes belong in chat replies, workflow scratch artifacts, or the user's own notes — **not** committed project docs.
- **`docs/` is reserved for documenting intentional source/code changes** (design notes for code, API behavior changes, algorithm/kernel modifications, PR-oriented change logs). Prefer short, code-linked notes over install diaries.
- Do not create files such as `docs/setup-*-report.md`, `docs/install-*-report.md`, or similar environment reports under this repo.

## Machine-local config (`.env`) — MANDATORY

Model checkpoint paths are **machine-local** and live in the **gitignored** `<repo>/.env` — never in tracked files. `project_config.py` loads it via python-dotenv (process env > `.env`; **no built-in defaults** — an unconfigured model id raises a helpful `KeyError`).

**Before running anything that loads model weights, an agent MUST:**

1. Ensure `.env` exists (else `cp .env.example .env` and fill in this node's paths; values may use `~`).
2. Run `python project_config.py --check <model_id>` and proceed only on exit 0. Report-all mode: `python project_config.py --check`.
3. Never commit machine-local absolute paths — to `project_config.py`, docs, or anywhere tracked. Node-specific paths go to `.env` only.

Keys: `MXMOE_MODEL_DS2`, `MXMOE_MODEL_MIXTRAL`, `MXMOE_MODEL_QWEN2_MOE`, `MXMOE_MODEL_QWEN2_MOE_57B` (mapping: `MODEL_ID2ENV` in `project_config.py`). Adding a new model id ⇒ extend `MODEL_ID2ENV` + `.env.example` + this list. Requires `python-dotenv` in the `mxmoe` env.

## Workflows (`.grok/workflows/`)

Project workflows are **Rhai orchestration scripts** under `.grok/workflows/`. They automate multi-step agent pipelines for this repo (env bring-up, optional package compile, smoke verification, later experiment stages). They are **not** part of the scientific code path; they exist so setup and repeatable checks can run in the background with a clear phase rail.

### Role in this project

| Purpose | Notes |
|---------|--------|
| Machine / env bring-up | Isolated conda env, CUDA toolchain, deps |
| Optional heavy installs | e.g. compile `flash_attn`, install `lm_eval` |
| Smoke gates | Imports, CUDA, kernel codegen/build/bench without inventing paper results |
| Ops reporting | Prefer **workflow run result + scratch files**; never install diaries under `docs/` |

### Current project workflows

| Name | Path | What it does |
|------|------|----------------|
| `setup-mxmoe-env` | `.grok/workflows/setup-mxmoe-env.rhai` | Create/use conda env `mxmoe`, install smoke Python deps (skip flash by default), init submodules / hadamard, arch prep, L0–L2 ending in `run_mxmoe_gg` FP16 bench |
| `install-flash-lmeval` | `.grok/workflows/install-flash-lmeval.rhai` | Compile-install `flash_attn` and install `lm_eval` into `mxmoe`, then verify FA2 + project `Evaluator` / lm_eval |
| `reproduce-acc` | `.grok/workflows/reproduce-acc.rhai` | **Accuracy-only** multi-GPU path for any `ID2NAME` MoE: required `model_id`+`model_path`; pins `gpu_a/b/c` so **calib A‖B** (and baseline on `gpu_c`) run in parallel; then ILP + MxMoE eval; **no** kernel/`run_mxmoe_gg` |

(User-level workflows under `~/.grok/workflows/` are separate and not listed here.)

### Rules when creating a **new** workflow

Every new workflow **must state its purpose clearly** before implementation is considered complete:

1. **File header comment** (top of the `.rhai`): 2–6 lines describing what the workflow does, what it does *not* do, and key side effects (env name, GPUs, network, long compile).
2. **`let meta = #{ ... }`** must include, as pure literals:
   - **`name`**: lowercase letters, digits, hyphens; matches the filename stem.
   - **`description`**: one sentence on *what* it does (user-visible).
   - **`when_to_use`**: one sentence on *when* to run it (trigger / situation).
   - **`phases`**: optional but recommended; titles must match `phase()` calls in the body.
3. **Args documentation**: in the header or a short block above arg parsing, list each arg, default, and meaning.
4. **Scope boundary**: explicitly say if it is env-only, install-only, experiment-only, or read-only; do not mix unrelated goals in one workflow without saying so.
5. **No silent `docs/` writes** for setup/install reports (see Documentation policy). Use `write_scratch_file` / `complete(...)` instead.
6. **Update this `AGENTS.md` table** when adding a project workflow under `.grok/workflows/`, so the catalog stays accurate.

When editing an existing workflow, keep `description` / `when_to_use` / header in sync with behavior changes.

## Single-GPU layer placement ("PP test", accelerate fit-map)

Load a MoE model that does **not** fit in one GPU (e.g. Qwen1.5-MoE-A2.7B bf16 ≈ 28.6GiB on a 24GB RTX 3090) for whole-model accuracy eval. Placement is done **only** by accelerate (`load_checkpoint_and_dispatch` + `device_map="auto"` + `max_memory`, optional `offload_folder` for disk spill) — no hand-written CPU↔GPU layer scheduler, no real pipeline parallelism (no pippy/microbatch/`torch.distributed`).

**Implementation** (all in `mxmoe/quant/moe_utils.py`):

- `build_max_memory(gpu_limit, cpu_limit)` → `max_memory` dict; GPU-only ⇒ CPU defaults to `128GiB` spill.
- `resolve_device_map(model_id, device_map, max_memory)` → explicit map wins; `max_memory` set ⇒ force `"auto"`; else legacy multi-GPU hand maps from `get_device_map`.
- `load_hf_model(..., max_memory=..., offload_folder=...)` → dispatches and prints the resulting `hf_device_map`. `no_split_module_classes` must be class **names** (accelerate compares `__class__.__name__`; passing classes silently disables it and splits layers mid-expert).

**Usage**:

```bash
# Full-precision PPL on one 3090 (keep ~8GiB free: 4k-token logits on the 151k vocab need ~5GiB)
CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.quant eval --model qwen2_moe --method rtn \
    --tasks ppl --max-gpu-memory 16GiB --max-cpu-memory 64GiB
# Gate tracing with the same placement flags
CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.moe_tracer --model qwen2_moe --trace_gate \
    --max-gpu-memory 8GiB --max-cpu-memory 64GiB [--offload-folder DIR]
```

**Rules / gotchas**:

- Never `.to()` the whole model or single layers after dispatch; hooks restore offloaded weights every forward. Offloaded modules sit on `meta` between forwards — read devices from runtime tensors (e.g. `logits.device`), never from `lm_head.weight.device`.
- Fit-map is **full-precision eval only**: it is rejected (SystemExit) together with `--qstr/--qconfig/--qweight`, because in-place quantization on offloaded layers is silently dropped. Calib `layer_out_norm` / online GPTQ keep their own CPU + per-layer path and do not use fit-map.
- Verified: `qwen2_moe` FP16 wikitext2 PPL = 6.79 (73×4096, ~5.2s/sample) on a single 3090 with `16GiB` GPU budget.

## Expert GPTQ export (W1–W4 A16) — status & usage

Uniform **weight-only GPTQ** on MoE expert linears (`gate` / `up` / `down` + shared experts; also dense MLP on non-MoE layers such as `ds2` layer 0). Attention / embed / norm / `lm_head` stay full precision. Entry point: **`python -m mxmoe.quant.export_gptq_hf`**.

### What is implemented

| Piece | Path / notes |
|-------|----------------|
| CLI + sequential expert GPTQ | `mxmoe/quant/export_gptq_hf.py` |
| GPTQ `fasterquant(..., return_quant_state=True)` | `mxmoe/quant/gptq.py` — returns `scale` / `zero` / `g_idx` / fake-quant `weight` |
| Pack helpers (GPTQ v1 layout) | `mxmoe/quant/gptq_pack.py` — `qweight` / `qzeros` / `scales` / `g_idx` |
| Export root (large HF dumps) | Default **`/mnt/raid/tzj/mxmoe/exported_hf_models/<name>/`**; override with env `MXMOE_EXPORT_HF_ROOT` or `--save` / `--save-name` |
| Remote-code copy for HF reload | After `save_pretrained`, copies `*.py` from the source checkpoint (e.g. `modeling_deepseek.py`) so `trust_remote_code` loads |

### Two export formats

| `--export-format` | Meaning | Bits | Load path |
|-------------------|---------|------|-----------|
| **`hf_float`** (default) | GPTQ-optimized **fake-quant float** weights; full HF `save_pretrained` tree | **1, 2, 3, 4, 8** | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` |
| **`gptq_pack`** | Packed **AutoGPTQ/v1** tensors (`qweight` / `qzeros` / `scales` / `g_idx`) in state_dict / safetensors | **2, 3, 4, 8 only** (W1 → `NotImplementedError`) | Needs a GPTQ backend (e.g. gptqmodel); **not** fully validated end-to-end in-repo |

Also written: `quantize_config.json`, `mxmoe_export_meta.json`, optional `eval_ppl.json` when `--eval-ppl`.

### Explicitly **not** implemented (raise `NotImplementedError`)

- Mixed-precision ILP `--qconfig` export
- Weight–activation schemes (e.g. W4A4) as export format
- **Online** Hadamard (`--online-had`); offline `--hadamard` is allowed for research but changes rotation/fuse story
- `--quant-attn` (attention GPTQ export)
- `gptq_pack` for **W1**

### Usage (conda env `mxmoe`)

```bash
# Check model path first
python project_config.py --check ds2

# Expert GPTQ W4A16 g128 asym → HF float dump on RAID (default root)
CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.export_gptq_hf \
  --model ds2 --wbits 4 --groupsize 128 \
  --nsamples 128 --seqlen 4096 \
  --device-map cpu \
  --export-format hf_float \
  --eval-ppl

# Other bits: --wbits 1|2|3|4
# Optional name: --save-name ds2-gptq-w4-g128-asym
# Optional pack (W2–W4/W8 only; needs GPTQ runtime to load):
#   --export-format gptq_pack

# Smoke: only first N layers
#   --max-layers 2
```

Load exported `hf_float` model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "/mnt/raid/tzj/mxmoe/exported_hf_models/ds2-gptq-w4-g128-asym"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)
```

### Verified once (ds2 / DeepSeek-V2-Lite, expert WkA16 g128 asym, `hf_float`)

WikiText2 PPL, `input_len=4096` (same evaluator path as `mxmoe.quant.evaluator`):

| Checkpoint | PPL (approx.) |
|------------|----------------|
| FP baseline | 5.92 |
| W4A16 GPTQ experts | 5.97 |
| W3A16 | 6.24 |
| W2A16 | 8.89 |
| W1A16 | ~1e5 (unusable quality without further recipes, e.g. Had) |

Artifacts (gitignored large dumps): under `MXMOE_EXPORT_HF_ROOT` / default RAID path above. Summary JSON may exist as `ds2_w1w2w3w4_summary.json` next to those dirs when produced by ops scripts — **do not** commit weight dumps.

### Agent notes

- Prefer **`hf_float`** for accuracy / reloadability; use **`gptq_pack`** only when a packed GPTQ consumer is required and bits ∈ {2,3,4,8}.
- Large exports belong on **RAID** (`/mnt/raid/tzj/...`), not `/mnt/data` (often full).
- Do not bake machine-local export paths into tracked code defaults beyond the documented RAID default; operators may override with `MXMOE_EXPORT_HF_ROOT`.

## Related conventions

- Prefer minimal, necessary source diffs; keep generated kernels/build trees out of git (see `.gitignore`).
- Prefer the isolated conda env **`mxmoe`** for MxMoE work; do not pollute unrelated envs unless the user asks.
- Large export trees under `outputs/` / RAID export roots are gitignored; do not commit model weights.
