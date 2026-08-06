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

## Related conventions

- Prefer minimal, necessary source diffs; keep generated kernels/build trees out of git (see `.gitignore`).
- Prefer the isolated conda env **`mxmoe`** for MxMoE work; do not pollute unrelated envs unless the user asks.
