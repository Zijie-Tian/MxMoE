# MxMoE project agent notes

## Unified environment — MANDATORY

- **Accuracy and performance use the same Conda environment `mxmoe`, configured with the performance/source-build stack.** This includes calibration, GPTQ, evaluation, all tests, CUDA kernels, and `3rdparty/vllm`; do not create a legacy accuracy environment or a separate `.venv`.
- Follow **[`doc/environment.md`](doc/environment.md)** for installation and verification. Root `requirements.txt` and `constraints.txt` define the shared Python dependencies; build Hadamard, FlashAttention, and local editable vLLM against that stack. Do not downgrade Torch/Transformers for accuracy work.
- Use `conda run --no-capture-output -n mxmoe python ...` or explicitly activate `mxmoe` before running commands. Never rely on whichever environment the shell inherited.
- Build native extensions and install dependencies with that Python (`python -m pip`), applying the root constraints. Never substitute `base`, `bitmoe`, another project environment, or system Python. Rebuild extensions after changing the Torch/CUDA ABI.
- Installation/import checks are not inference or performance validation. Report kernel execution, `LLM.generate`, and benchmark results separately; never infer them from a successful build.

## Documentation policy (`doc/`, `docs/`)

- **`doc/environment.md` is the user-requested, maintained environment configuration guide.** Keep it aligned with dependency files and installation workflows; document the single shared environment, not accuracy/performance alternatives.
- Keep machine-local checkpoint paths, credentials, build-cache locations, installation logs, and per-machine reports out of tracked documentation. Use `.env` for model paths and shell variables/scratch artifacts for build configuration and logs.
- Other `docs/` content remains reserved for intentional source/code changes. Do not add setup/install diaries or duplicate environment runbooks.

## Machine-local config (`.env`) — MANDATORY

Model checkpoint paths are **machine-local** and live in the **gitignored** `<repo>/.env` — never in tracked files. `project_config.py` loads it via python-dotenv (process env > `.env`; **no built-in defaults** — an unconfigured model id raises a helpful `KeyError`).

**Before running anything that loads model weights, an agent MUST:**

1. Ensure `.env` exists (else `cp .env.example .env` and fill in this node's paths; values may use `~`).
2. Run `python project_config.py --check <model_id>` and proceed only on exit 0. Report-all mode: `python project_config.py --check`.
3. Never commit machine-local absolute paths — to `project_config.py`, docs, or anywhere tracked. Node-specific paths go to `.env` only.

Keys: `MXMOE_MODEL_DS2`, `MXMOE_MODEL_MIXTRAL`, `MXMOE_MODEL_QWEN2_MOE`, `MXMOE_MODEL_QWEN2_MOE_57B` (mapping: `MODEL_ID2ENV` in `project_config.py`). Adding a new model id ⇒ extend `MODEL_ID2ENV` + `.env.example` + this list. Requires `python-dotenv` in the `mxmoe` env.

## Workflows (`.grok/workflows/`)

Project workflows are **Rhai orchestration scripts** under `.grok/workflows/`. They automate shared-environment bring-up, extension builds, smoke verification, and later experiments. They are **not** part of the scientific code path; all use the same `mxmoe` environment documented above.

### Role in this project

| Purpose | Notes |
|---------|--------|
| Machine / env bring-up | Unified `mxmoe` performance stack for accuracy and performance |
| Extension install / rebuild | Hadamard, FlashAttention, source-editable vLLM; shared accuracy dependencies |
| Smoke gates | Imports, CUDA, kernel codegen/build/bench without inventing paper results |
| Ops reporting | Prefer **workflow run result + scratch files**; never install diaries under `docs/` |

### Current project workflows

| Name | Path | What it does |
|------|------|----------------|
| `setup-mxmoe-env` | `.grok/workflows/setup-mxmoe-env.rhai` | Create/reuse unified `mxmoe`, install root requirements, build/reuse compatible extensions and source-editable vLLM, then L0–L2 ending in `run_mxmoe_gg` FP16 bench |
| `install-flash-lmeval` | `.grok/workflows/install-flash-lmeval.rhai` | Install/reuse compatible FlashAttention and lm_eval within the same constrained stack, then verify FA2 + project `Evaluator` / lm_eval |
| `reproduce-acc` | `.grok/workflows/reproduce-acc.rhai` | **Accuracy-only workload, shared environment**: required `model_id`+`model_path`; pins `gpu_a/b/c` for parallel calibration and baseline, then ILP + MxMoE eval; no kernel benchmark |
| `quant-lowbit-gptq-had` | `.grok/workflows/quant-lowbit-gptq-had.rhai` | **Quant-only workload, shared environment**: GPTQ-Had for 12 solved `ds2` / `qwen2_moe` qconfigs (BPV 2.25/3.25, R=0/0.5/1); one GPU per model, six configs sequential; no ILP, calib, PPL, or kernel benchmark |

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

## Native grouped GEMM benchmark (F16 / W8A8 / W4A4)

This is an **operator-only benchmark** with synthetic inputs and workload shapes, not model inference, PPL, or vLLM. It does not load checkpoints. Run each case on a single GPU, without TP, in the shared `mxmoe` environment. For RTX 3090 comparisons, use CUDA 12.8 and SM86 consistently; do not mix results from an older CUDA 12.1 build.

### Implementation boundaries

- `mxmoe/kernels/src/ref_impl.cuh`: standard CUTLASS wrappers `group_gemm_cutlass_f16`, `group_gemm_cutlass_w8a8`, and `group_gemm_cutlass_w4a4` in `mxmoe_ref`. Do not reintroduce separate W8A8/W4A4 implementation files or put operator implementations into the test driver.
- `mxmoe/kernels/src/test.cu`: CLI, workload parsing, existing F16/MxMoE driver, and the shared `run_cutlass_integer<Bits>` / `run_cutlass_integer_group<Bits>` driver. W4A4 and W8A8 reuse `check_or_bench_group_gemm` and `bench_func`; do not duplicate their timing or checking loops.
- `mxmoe/kernels/src/test_utils.h`: test-data generation, INT4 packing, integer reference calculation, and shared comparison/timing helpers. Integer outputs require exact comparison, not conversion to FP32 or a floating tolerance.
- Refactor only project-owned kernel code when working on this benchmark; do not refactor unrelated code or submodules, including vendored CUTLASS and vLLM. Leave experiment changes uncommitted unless explicitly requested.

### Build and invocation

Follow `doc/environment.md` for the environment and host compiler. Keep local paths in shell variables: `MXMOE_KERNEL_BUILD` is the configured native build directory, and workload variables below name JSON or legacy text input files. Configure the native CMake project at `mxmoe/kernels/src` with `-DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"`, `-DCMAKE_CUDA_ARCHITECTURES=86`, and `-DCMAKE_BUILD_TYPE=Release` for SM86. Clear inherited `NVCC_PREPEND_FLAGS` / `NVCC_APPEND_FLAGS` when they refer to another environment.

The `test` target still links generated MxMoE kernels. `-DGENERATED_DIR=...` selects their source directory; an integer-only comparison build may contain only generated F16 kernels. To measure MxMoE W4A4/W8A8, use a build containing those actual registered candidates, not an unmatched filter that produces `N/A`.

```bash
# Run from the repository root after configuring the shared environment/build.
: "${MXMOE_KERNEL_BUILD:?Set the configured native build directory}"
env -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS conda run --no-capture-output -n mxmoe \
  cmake --build "$MXMOE_KERNEL_BUILD" --target test -j 8

# Built-in integer correctness checks (no workload file required).
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" check --backend cutlass_w8a8
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" check --backend cutlass_w4a4

# Exact checks on a small supplied workload; use bench to time larger workloads.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" check --backend cutlass_w4a4 --input "$W4A4_WORKLOAD"
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" bench --backend cutlass_w4a4 --input "$W4A4_WORKLOAD"
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" bench --backend cutlass_w8a8 --input "$W8A8_WORKLOAD"

# Existing F16 reference path. The filter excludes generated MxMoE candidates;
# CUTLASS F16 candidates still run. There is no cutlass_f16 CLI backend.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" check --input "$F16_WORKLOAD" --filter cutlass

# Requires generated W4A4 MxMoE candidates in this build; use w8a8 for W8A8.
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  "$MXMOE_KERNEL_BUILD/test" bench --input "$W4A4_WORKLOAD" --filter groupgemm_hz_fused_w4a4
```

### Workloads and correctness

The integer backends require every problem to match the selected W/A bits, `gsize=-1`, and `sym=true`. Dimensions must be positive; empty experts are rejected. W8A8 requires `K % 16 == 0`, W4A4 requires `K % 32 == 0`, and both require `N % 4 == 0`. A is row-major `[M,K]`; B is column-major `[K,N]`, equivalently row-major `[N,K]` weights. INT4 stores the even element in the low nibble and the odd element in the high nibble. Leading dimensions count logical elements, not compressed bytes.

Example small W4A4 JSON, covering multiple K iterations and the `K=1408` tail:

```json
{
  "layer-1": {
    "gate_up": [
      {"shape": [17, 128, 1408], "w_bits": 4, "a_bits": 4, "gsize": -1, "sym": true},
      {"shape": [17, 128, 2816], "w_bits": 4, "a_bits": 4, "gsize": -1, "sym": true}
    ]
  }
}
```

For W8A8, change both bit fields to 8. Legacy text is also accepted, e.g. `4x4x-1xsym 17x128x1408` (one problem per line). Workload files specify shapes/precision, not tensor values; the driver generates synthetic integers.

`check` without `--input` validates only built-ins. With `--input`, empty inputs or any group beyond the host-reference limit (`sum(2*M*N*K) > 20,000,000`) return failure; built-in success must not be reported as supplied-workload correctness. Preserve the W8A8 exact accumulator case `33032065` and W4A4 non-periodic data, signed boundaries, and independent packing checks.

### Timing and comparison rules

Integer `bench` uses 30 warmups and 50 CUDA-event measurements, reporting the median as `median_ms`. Timing surrounds the grouped host launcher, including per-call initialization; input generation, allocation, packing, and reference calculation are outside the measured region. Results are printed to stdout; the integer backends do not use `--output` or `--filter`. The existing default-path CSV field `avg_time` also means median, not arithmetic mean.

CUTLASS W4A4/W8A8 write **INT32 without dequantization**; MxMoE writes **FP16 with fused dequantization**. The standard integer wrappers each use a fixed tile, while MxMoE may select the fastest candidate. Disclose these differences rather than treating the ratio as a pure MMA speedup. For matched measurements, keep workload shapes, routing, GPU, CUDA version, and timing boundaries fixed, and record the candidate set. Aggregate throughput as total nominal FLOPs divided by summed median seconds; do not average per-case TFLOPS. Keep logs/results in ignored scratch directories, not tracked documentation or `.planning/` commits.

## Related conventions

- Prefer minimal, necessary source diffs; keep generated kernels/build trees out of git (see `.gitignore`).
- Use the isolated Conda environment **`mxmoe`** for MxMoE work; all repository tests MUST follow the mandatory environment rule above.
- Large export trees under `outputs/` / RAID export roots are gitignored; do not commit model weights.
