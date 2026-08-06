# MxMoE project agent notes

## Documentation policy (`docs/`)

- **Do not** use `docs/` (or other in-repo markdown) to record environment setup, package installs, conda/CUDA configuration, machine-local paths, or similar ops/runbooks.
- Those notes belong in chat replies, workflow scratch artifacts, or the user's own notes — **not** committed project docs.
- **`docs/` is reserved for documenting intentional source/code changes** (design notes for code, API behavior changes, algorithm/kernel modifications, PR-oriented change logs). Prefer short, code-linked notes over install diaries.
- Do not create files such as `docs/setup-*-report.md`, `docs/install-*-report.md`, or similar environment reports under this repo.

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
| `reproduce-acc` | `.grok/workflows/reproduce-acc.rhai` | **Accuracy-only** paper path for **any** `ID2NAME` MoE model: **required** `model_id` + `model_path` args; gate, gptq-had calib, ILP, Baseline + MxMoE PPL and optional zero-shot; **no** kernel/`run_mxmoe_gg` |

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

## Related conventions

- Prefer minimal, necessary source diffs; keep generated kernels/build trees out of git (see `.gitignore`).
- Prefer the isolated conda env **`mxmoe`** for MxMoE work; do not pollute unrelated envs unless the user asks.
