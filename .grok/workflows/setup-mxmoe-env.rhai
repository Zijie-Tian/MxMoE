// setup-mxmoe-env: Create an isolated conda env for MxMoE and verify with L0/L1/L2 smokes.
// Final gate: FP16 GroupGEMM generate → cmake build → bench (run_mxmoe_gg) without HF model weights.
let meta = #{
    name: "setup-mxmoe-env",
    description: "Create a project-local conda env for MxMoE, apply 3090-friendly patches, and verify with L0/L1/L2 smokes ending in run_mxmoe_gg FP16 bench",
    when_to_use: "On a new machine/node when you need a fresh conda env that can build and bench MxMoE kernels",
    phases: [
        #{ title: "Inventory", detail: "GPU, CUDA, conda, disk, repo state" },
        #{ title: "Conda-Env", detail: "create env, install torch + smoke requirements" },
        #{ title: "Repo-Prep", detail: "submodules, hadamard, arch patches, synthetic gate trace" },
        #{ title: "Verify-L0-L1", detail: "Python imports + fast-hadamard CUDA" },
        #{ title: "Verify-L2", detail: "run_mxmoe_gg FP16 layer smoke (final gate)" },
        #{ title: "Report", detail: "write setup report artifact" },
    ],
};

// ---------------------------------------------------------------------------
// Args (all optional)
//   env_name       default "mxmoe"
//   python         default "3.12"
//   cuda_home      default "" (agent auto-detects /usr/local/cuda*)
//   batch_size     default 512  (L2 tokens)
//   layer          default 11
//   skip_flash     default true
//   force_recreate default false  (conda remove -n env --all if exists)
//   dry_run        default false  (inventory + plan only, no install)
// ---------------------------------------------------------------------------

fn arg_str(a, key, default_val) {
    if a == () { return default_val; }
    let v = a[key];
    if v == () { return default_val; }
    if type_of(v) != "string" { return default_val; }
    if v == "" { return default_val; }
    v
}

fn arg_bool(a, key, default_val) {
    if a == () { return default_val; }
    let v = a[key];
    if v == () { return default_val; }
    if type_of(v) == "bool" { return v; }
    if type_of(v) == "string" {
        if v == "true" || v == "1" || v == "yes" { return true; }
        if v == "false" || v == "0" || v == "no" { return false; }
    }
    default_val
}

fn arg_int(a, key, default_val) {
    if a == () { return default_val; }
    let v = a[key];
    if v == () { return default_val; }
    if type_of(v) == "i64" || type_of(v) == "int" { return v; }
    if type_of(v) == "f64" { return v; }
    if type_of(v) == "string" {
        // leave as string for prompt interpolation if needed
        return v;
    }
    default_val
}

let env_name = arg_str(args, "env_name", "mxmoe");
let py_ver = arg_str(args, "python", "3.12");
let cuda_home = arg_str(args, "cuda_home", "");
let batch_size = arg_int(args, "batch_size", 512);
let layer = arg_int(args, "layer", 11);
let skip_flash = arg_bool(args, "skip_flash", true);
let force_recreate = arg_bool(args, "force_recreate", false);
let dry_run = arg_bool(args, "dry_run", false);

let inventory_schema = #{
    "type": "object",
    "required": ["ok", "gpu_name", "compute_capability", "cuda_home", "conda_ok", "disk_free_gb", "repo_root", "notes"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "gpu_name": #{ "type": "string" },
        "compute_capability": #{ "type": "string" },
        "cuda_home": #{ "type": "string" },
        "nvcc_version": #{ "type": "string" },
        "cmake_version": #{ "type": "string" },
        "conda_ok": #{ "type": "boolean" },
        "disk_free_gb": #{ "type": "number" },
        "repo_root": #{ "type": "string" },
        "submodules_initialized": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let step_schema = #{
    "type": "object",
    "required": ["ok", "summary", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "details": #{ "type": "string" },
        "env_path": #{ "type": "string" },
        "log_tail": #{ "type": "string" },
    },
};

let l2_schema = #{
    "type": "object",
    "required": ["ok", "summary", "exit_code", "generated_ok", "binary_ok", "bench_ok", "bench_path", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "exit_code": #{ "type": "integer" },
        "generated_ok": #{ "type": "boolean" },
        "binary_ok": #{ "type": "boolean" },
        "bench_ok": #{ "type": "boolean" },
        "bench_path": #{ "type": "string" },
        "details": #{ "type": "string" },
    },
};

// =============================================================================
phase("Inventory");
// =============================================================================

let inv_prompt = "";
inv_prompt += "You are setting up MxMoE on this machine. Work in the CURRENT workspace (MxMoE repo root).\n";
inv_prompt += "Goal: inventory hardware/toolchain so later steps can build an isolated conda env and run kernel smoke tests.\n\n";
inv_prompt += "Run shell commands (do not guess):\n";
inv_prompt += "1) pwd; git rev-parse --show-toplevel; git branch --show-current\n";
inv_prompt += "2) nvidia-smi (note GPU name, driver, CUDA version)\n";
inv_prompt += "3) Detect compute capability (e.g. deviceQuery or python with pynvml/torch if available; else map RTX 3090→86, 4090→89, A100→80)\n";
inv_prompt += "4) which conda; conda --version; conda env list\n";
inv_prompt += "5) which nvcc; nvcc --version; echo CUDA_HOME; ls /usr/local/cuda* 2>/dev/null | head\n";
inv_prompt += "6) cmake --version; ninja --version; g++ --version | head -1\n";
inv_prompt += "7) df -h . | tail -1 (disk free)\n";
inv_prompt += "8) git submodule status | head -20; test whether mxmoe/3rdparty/fast-hadamard-transform has files beyond empty dir\n";
inv_prompt += "9) ls perf/performance_table.json; ls calib 2>/dev/null; test -f project_config.py\n\n";
inv_prompt += "If args suggested cuda_home is non-empty, prefer it: " + cuda_home + "\n";
inv_prompt += "Return structured fields. Set ok=false only if no GPU or no conda or disk free < 20GB.\n";
inv_prompt += "Put blockers/warnings in notes.\n";

let inv = agent(inv_prompt, #{
    label: "inventory",
    capability_mode: "all",
    output_schema: inventory_schema,
});

if inv == () || !inv.success || inv.output == () {
    pause("infra", "Inventory agent failed. Resume after fixing shell/GPU access, or re-run the workflow.");
}

if inv.output.ok != true {
    let msg = "Inventory reported not-ok. notes=" + inv.output.notes;
    pause("infra", msg);
}

log("Inventory OK: GPU=" + inv.output.gpu_name + " cc=" + inv.output.compute_capability + " cuda_home=" + inv.output.cuda_home);

if dry_run {
    let dry_report = "";
    dry_report += "# MxMoE env setup (dry_run)\n\n";
    dry_report += "- env_name: " + env_name + "\n";
    dry_report += "- python: " + py_ver + "\n";
    dry_report += "- skip_flash: " + skip_flash.to_string() + "\n";
    dry_report += "- L2 target: run_mxmoe_gg FP16 bs=" + batch_size.to_string() + " layer=" + layer.to_string() + "\n\n";
    dry_report += "## Inventory\n";
    dry_report += inv.output.notes + "\n";
    dry_report += "\nGPU: " + inv.output.gpu_name + " cc=" + inv.output.compute_capability + "\n";
    dry_report += "Would proceed to conda create + L0/L1/L2 if dry_run=false.\n";
    let p = write_scratch_file("setup-mxmoe-env-dry-run.md", dry_report);
    complete(#{
        summary: "dry_run only: inventory passed; no install performed",
        dry_run: true,
        report: p,
        final_test_target: "python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string(),
    });
}

// =============================================================================
phase("Conda-Env");
// =============================================================================

let conda_prompt = "";
conda_prompt += "Create/update an ISOLATED conda environment for the MxMoE repo and install Python deps.\n";
conda_prompt += "Repo root: " + inv.output.repo_root + "\n";
conda_prompt += "Env name: " + env_name + "\n";
conda_prompt += "Python: " + py_ver + "\n";
conda_prompt += "CUDA_HOME to export for builds: " + inv.output.cuda_home + "\n";
conda_prompt += "skip_flash_attn: " + skip_flash.to_string() + "\n";
conda_prompt += "force_recreate: " + force_recreate.to_string() + "\n\n";
conda_prompt += "Steps (execute, do not skip):\n";
conda_prompt += "1) source conda.sh if needed so `conda activate` works in non-interactive shells.\n";
conda_prompt += "2) If force_recreate and env exists: conda env remove -n " + env_name + " -y\n";
conda_prompt += "3) If env missing: conda create -n " + env_name + " python=" + py_ver + " -y\n";
conda_prompt += "4) Activate env. Export CUDA_HOME, PATH, LD_LIBRARY_PATH from the inventory cuda_home.\n";
conda_prompt += "5) From repo root, create requirements-smoke.txt from requirements.txt:\n";
conda_prompt += "   - DROP the line flash_attn==...\n";
conda_prompt += "   - ADD regex==2024.11.6 if not present (compose_kernel imports regex)\n";
conda_prompt += "   - Keep other pins.\n";
conda_prompt += "6) Install torch first if needed: pip install torch==2.6.0 matching cu124 when possible\n";
conda_prompt += "   (pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124) OR pip install -r requirements-smoke.txt\n";
conda_prompt += "7) pip install -r requirements-smoke.txt (retry once on transient network errors).\n";
conda_prompt += "8) Verify: python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"\n";
conda_prompt += "   Must show True and a GPU name.\n";
conda_prompt += "9) pip install regex jsbeautifier if missing after requirements.\n\n";
conda_prompt += "Do NOT install flash_attn unless skip_flash is false.\n";
conda_prompt += "Do NOT download HF models.\n";
conda_prompt += "Return ok=true only if torch.cuda.is_available() is True inside the env.\n";
conda_prompt += "Put env prefix path in env_path. Include last 40 lines of install log in log_tail if long.\n";

let conda_r = agent(conda_prompt, #{
    label: "conda-install",
    capability_mode: "all",
    output_schema: step_schema,
});

if conda_r == () || !conda_r.success || conda_r.output == () || conda_r.output.ok != true {
    let d = "conda step failed";
    if conda_r != () && conda_r.success && conda_r.output != () {
        d = conda_r.output.summary + " | " + conda_r.output.details;
    }
    pause("infra", "Conda env install failed: " + d + ". Fix network/CUDA pip issues then resume or re-run.");
}

log("Conda env ready: " + conda_r.output.summary);

// =============================================================================
phase("Repo-Prep");
// =============================================================================

let prep_prompt = "";
prep_prompt += "Prepare the MxMoE repo for kernel smoke on THIS node. Repo: " + inv.output.repo_root + "\n";
prep_prompt += "Conda env name: " + env_name + " (activate it for all pip/python).\n";
prep_prompt += "Detected compute capability string for tile tables: " + inv.output.compute_capability + "\n";
prep_prompt += "CUDA_HOME: " + inv.output.cuda_home + "\n\n";
prep_prompt += "Execute ALL of the following:\n\n";
prep_prompt += "### A) Submodules\n";
prep_prompt += "cd repo && git submodule update --init --recursive\n";
prep_prompt += "Confirm mxmoe/3rdparty/fast-hadamard-transform has setup.py and mxmoe/kernels/src/3rdparty/{cutlass,cccl,fmt,pybind11} are non-empty.\n\n";
prep_prompt += "### B) Install fast-hadamard-transform into the env\n";
prep_prompt += "conda activate " + env_name + "\n";
prep_prompt += "export CUDA_HOME=" + inv.output.cuda_home + "\n";
prep_prompt += "cd mxmoe/3rdparty/fast-hadamard-transform && pip install . && cd -\n";
prep_prompt += "python -c \"import fast_hadamard_transform; print('hadamard-ok')\"\n\n";
prep_prompt += "### C) GPU arch patches (required for non-4090 nodes)\n";
prep_prompt += "1) mxmoe/kernels/src/CMakeLists.txt: set(CMAKE_CUDA_ARCHITECTURES XX) where XX matches this GPU.\n";
prep_prompt += "   Use " + inv.output.compute_capability + " if it looks like 80/86/89; for RTX 3090 use 86; 4090 use 89; A100 use 80.\n";
prep_prompt += "2) mxmoe/kernels/tile_config.py get_possible_tile_list: arch_info and tile_list only have '80' and '89'.\n";
prep_prompt += "   If compute_capability is '86' (or other missing key), ALIAS it to the closest table:\n";
prep_prompt += "   - prefer mapping 86 -> reuse 89 tile entries (copy key or fallback lookup)\n";
prep_prompt += "   Make get_possible_tile_list not KeyError on this machine's get_gpu_info()['cc'].\n";
prep_prompt += "   Minimal surgical edit; do not rewrite the whole file.\n\n";
prep_prompt += "### D) performance_table path convenience\n";
prep_prompt += "mkdir -p calib/perf\n";
prep_prompt += "If perf/performance_table.json exists and calib/perf/ one does not, copy it.\n\n";
prep_prompt += "### E) Synthetic gate trace for L2 (NO model download)\n";
prep_prompt += "Create calib/gate/qwen2_moe/wiki2/4096/moe-gate.json with schema used by mxmoe/kernels/gen_workload.py:\n";
prep_prompt += "  topk=4, NK=[1408,2048] (or [2816,2048] if you confirm from code comments),\n";
prep_prompt += "  num_shared_experts=4, num_tokens=4096,\n";
prep_prompt += "  layer-" + layer.to_string() + " with access_freq length 60 positive ints summing >0,\n";
prep_prompt += "  weights_sum length 60 floats, percentile_stats={}.\n";
prep_prompt += "Also ensure MODEL_ID_TO_TRACE_FILE path exists (gen_workload defaults).\n\n";
prep_prompt += "### F) Model paths live in the gitignored repo-root .env (MXMOE_MODEL_* keys; see .env.example). Do NOT edit tracked project_config.py; L2 should not load HF weights, so .env model keys are not required here.\n";
prep_prompt += "Do NOT download models.\n\n";
prep_prompt += "Return ok=true only if submodules ok, hadamard imports, arch patch applied (or already correct for 89), and gate json exists.\n";

let prep_r = agent(prep_prompt, #{
    label: "repo-prep",
    capability_mode: "all",
    output_schema: step_schema,
});

if prep_r == () || !prep_r.success || prep_r.output == () || prep_r.output.ok != true {
    let d = "repo-prep failed";
    if prep_r != () && prep_r.success && prep_r.output != () {
        d = prep_r.output.summary + " | " + prep_r.output.details;
    }
    pause("infra", "Repo prep failed: " + d);
}

log("Repo prep OK: " + prep_r.output.summary);

// =============================================================================
phase("Verify-L0-L1");
// =============================================================================

let v01_prompt = "";
v01_prompt += "Run L0 and L1 environment verification for MxMoE. Activate conda env `" + env_name + "` first.\n";
v01_prompt += "Repo: " + inv.output.repo_root + "\n";
v01_prompt += "export CUDA_HOME=" + inv.output.cuda_home + "\n\n";
v01_prompt += "### L0 — imports + CUDA\n";
v01_prompt += "Run a Python script that asserts:\n";
v01_prompt += "- sys.version_info[:2] matches 3.12 (or the env python major.minor)\n";
v01_prompt += "- import torch; torch.cuda.is_available()\n";
v01_prompt += "- import transformers, datasets, accelerate, einops, regex, jsbeautifier\n";
v01_prompt += "- import pynvml, pycuda.driver (pycuda.autoinit may init CUDA)\n";
v01_prompt += "- import gurobipy  (import only; license not required)\n";
v01_prompt += "- from mxmoe.kernels.tile_config import get_gpu_info\n";
v01_prompt += "- from mxmoe.kernels.compose_kernel import TemplateGenerator, KernelType\n";
v01_prompt += "- print get_gpu_info() and ensure 'cc' is present\n";
v01_prompt += "Print exactly: L0 PASS\n\n";
v01_prompt += "### L1 — fast_hadamard_transform on GPU\n";
v01_prompt += "import fast_hadamard_transform; x=torch.randn(8,1024,device='cuda',dtype=torch.float16);\n";
v01_prompt += "y=fast_hadamard_transform.hadamard_transform(x); print shapes; print L1 PASS\n\n";
v01_prompt += "Return ok=true only if both L0 PASS and L1 PASS appear in command output.\n";
v01_prompt += "Put full relevant stdout in details.\n";

let v01 = agent(v01_prompt, #{
    label: "verify-l0-l1",
    capability_mode: "all",
    output_schema: step_schema,
});

if v01 == () || !v01.success || v01.output == () || v01.output.ok != true {
    let d = "L0/L1 failed";
    if v01 != () && v01.success && v01.output != () {
        d = v01.output.summary + " | " + v01.output.details;
    }
    pause("verification", "L0/L1 smoke failed: " + d);
}

log("L0/L1 PASS");

// =============================================================================
phase("Verify-L2");
// =============================================================================

let l2_prompt = "";
l2_prompt += "FINAL GATE for MxMoE env setup. This is the acceptance test for the environment.\n\n";
l2_prompt += "Repo: " + inv.output.repo_root + "\n";
l2_prompt += "Activate conda env: " + env_name + "\n";
l2_prompt += "export CUDA_HOME=" + inv.output.cuda_home + "\n";
l2_prompt += "export PATH=$CUDA_HOME/bin:$PATH\n\n";
l2_prompt += "### Final command (must run from repo root)\n";
l2_prompt += "python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string() + "\n";
l2_prompt += "(No --qconfig / --qstr → FP16 path. Does NOT require HF model weights if synthetic gate trace exists.)\n\n";
l2_prompt += "This script will: gen_workload → TemplateGenerator → cmake -B build -G Ninja → build → ./build/test bench\n\n";
l2_prompt += "### Pass criteria (ALL required)\n";
l2_prompt += "1) process exit code 0\n";
l2_prompt += "2) generated_ok: mxmoe/kernels/src/generated/ has hz_fused_decl.cuh and at least one .cu\n";
l2_prompt += "3) binary_ok: mxmoe/kernels/src/build/test exists and is executable\n";
l2_prompt += "4) bench_ok: out/bench/ has a new file for this model/dataset/bs (or the --output path the script used)\n";
l2_prompt += "5) No CUDA error 'no kernel image available' / illegal instruction\n\n";
l2_prompt += "If cmake fails due to empty generated/, re-run TemplateGenerator path or fix arch alias and retry ONCE.\n";
l2_prompt += "If OOM, retry once with smaller bs=128 by re-invoking run_mxmoe_gg with --bs 128 (note in details).\n\n";
l2_prompt += "Set ok=true only if all pass criteria hold.\n";
l2_prompt += "Fill exit_code, generated_ok, binary_ok, bench_ok, bench_path, details (stdout/stderr tail).\n";

let l2 = agent(l2_prompt, #{
    label: "verify-l2-gg",
    capability_mode: "all",
    output_schema: l2_schema,
});

if l2 == () || !l2.success || l2.output == () {
    pause("verification", "L2 agent failed to return a structured result. Inspect run logs and resume/re-run.");
}

if l2.output.ok != true {
    let msg = "L2 FINAL GATE FAILED: " + l2.output.summary + " | " + l2.output.details;
    pause("verification", msg);
}

log("L2 PASS: " + l2.output.summary);

// =============================================================================
phase("Report");
// =============================================================================

let report_prompt = "";
report_prompt += "Write a concise markdown setup report for the MxMoE conda environment on this node.\n";
report_prompt += "Use facts from the completed steps (do not invent success):\n\n";
report_prompt += "Inventory notes: " + inv.output.notes + "\n";
report_prompt += "GPU: " + inv.output.gpu_name + " cc=" + inv.output.compute_capability + "\n";
report_prompt += "CUDA_HOME: " + inv.output.cuda_home + "\n";
report_prompt += "Conda summary: " + conda_r.output.summary + "\n";
report_prompt += "Conda details: " + conda_r.output.details + "\n";
report_prompt += "Repo prep: " + prep_r.output.summary + "\n";
report_prompt += "L0/L1: " + v01.output.summary + "\n";
report_prompt += "L2 summary: " + l2.output.summary + "\n";
report_prompt += "L2 bench_path: " + l2.output.bench_path + "\n";
report_prompt += "env_name: " + env_name + "\n";
report_prompt += "skip_flash: " + skip_flash.to_string() + "\n\n";
report_prompt += "Report MUST include sections:\n";
report_prompt += "1. Environment (conda name, python, torch, CUDA)\n";
report_prompt += "2. Patches applied (CMake arch, tile_config alias, synthetic trace path)\n";
report_prompt += "3. Verification results (L0/L1/L2)\n";
report_prompt += "4. Final test target command that passed\n";
report_prompt += "5. How to activate on this node: conda activate " + env_name + "\n";
report_prompt += "6. Known non-goals (no flash_attn, no Gurobi license, no HF model, no bits_solver)\n";
report_prompt += "7. Next steps for real experiments (download Qwen1.5-MoE, set MXMOE_MODEL_QWEN2_MOE in .env, optional Gurobi)\n\n";
report_prompt += "Do NOT write under repo docs/ (project AGENTS.md: no install/env reports in docs/).\n";
report_prompt += "Return the full markdown report text in details, and ok=true. Optionally note scratch-only delivery.\n";

let rep = agent(report_prompt, #{
    label: "write-report",
    capability_mode: "all",
    output_schema: step_schema,
});

let report_body = "";
if rep != () && rep.success && rep.output != () && rep.output.details != () {
    report_body = rep.output.details;
}
let report_path = write_scratch_file("setup-mxmoe-env-report.md", report_body);

let final_cmd = "python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string();

let scratch = "";
scratch += "# MxMoE setup-mxmoe-env result\n\n";
scratch += "**Final test target (PASSED):** `" + final_cmd + "`\n\n";
scratch += "- env: " + env_name + "\n";
scratch += "- GPU: " + inv.output.gpu_name + " (cc " + inv.output.compute_capability + ")\n";
scratch += "- L2 bench: " + l2.output.bench_path + "\n";
scratch += "- report: " + report_path + "\n";
let sp = write_scratch_file("setup-mxmoe-env-result.md", scratch);

complete(#{
    summary: "MxMoE conda env ready; L0/L1/L2 passed",
    env_name: env_name,
    gpu: inv.output.gpu_name,
    compute_capability: inv.output.compute_capability,
    final_test_target: final_cmd,
    final_test_meaning: "FP16 GroupGEMM path: synthetic/real gate trace → template kernel codegen → CMake/NVCC build → ./build/test bench. Proves Python deps, CUDA toolchain, submodules, and arch patches without downloading MoE weights.",
    bench_path: l2.output.bench_path,
    report_path: report_path,
    scratch: sp,
    l2_ok: true,
});
