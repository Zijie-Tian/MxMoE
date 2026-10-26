// setup-mxmoe-env: Create or reuse the one conda env `mxmoe` for every MxMoE
// accuracy, calibration, GPTQ, test, native build, and vLLM performance job.
// Install, reuse, and native builds execute doc/environment.md. Do not keep a
// second copy of its source-cache pins, FA3 patch, or machine build paths.
// Does not install torch 2.6/cu124, download HF weights, or run vLLM LLM.generate.
// Final gate remains FP16 GroupGEMM generate → cmake build → bench (run_mxmoe_gg) without HF model weights.
//
// Args (all optional):
//   env_name       default "mxmoe" (any other value is rejected)
//   python         default "3.12" (doc/environment.md create spec; 3.12.13 also accepted)
//   cuda_home      default "" (this arg if it is CUDA 12.8, else discover a
//                  12.8 toolkit; never inherited 12.4 or newest cuda*)
//   batch_size     default 512  (L2 tokens)
//   layer          default 11
//   skip_flash     default true (reuse an already verified compatible
//                  FlashAttention build; never omit it or skip its check)
//   force_recreate default false  (conda remove -n mxmoe --all if it exists)
//   dry_run        default false  (read-only inventory + plan, no install)
// ---------------------------------------------------------------------------

let meta = #{
    name: "setup-mxmoe-env",
    description: "Create or reuse conda env mxmoe on CUDA 12.8, install requirements.txt under constraints.txt, reuse or build Hadamard, FlashAttention, and source-editable vLLM, then pass L0/L1/L2 ending in run_mxmoe_gg FP16 bench",
    when_to_use: "On a new machine or when the single mxmoe performance stack must be created or verified before kernel smoke or later experiments",
    phases: [
        #{ title: "Inventory", detail: "GPU, CUDA 12.8, gcc-13, conda, disk, repo state (read-only)" },
        #{ title: "Conda-Env", detail: "init submodules, then execute doc/environment.md section 3 Python install" },
        #{ title: "Repo-Prep", detail: "submodules, then the guide's Hadamard, FlashAttention, and editable vLLM steps" },
        #{ title: "Verify-L0-L1", detail: "guide section 5 import check and small CUDA numeric smoke" },
        #{ title: "Verify-L2", detail: "run_mxmoe_gg FP16 layer smoke (final gate)" },
        #{ title: "Report", detail: "write setup report artifact" },
    ],
};

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

if env_name != "mxmoe" {
    pause("infra", "Refusing env_name='" + env_name + "'. The only supported conda environment is mxmoe.");
}
if py_ver != "3.12" && py_ver != "3.12.13" {
    pause("infra", "Refusing python='" + py_ver + "'. doc/environment.md creates mxmoe with python=3.12.");
}

let inventory_schema = #{
    "type": "object",
    "required": ["ok", "gpu_name", "compute_capability", "cuda_home", "cuda_release", "conda_ok", "disk_free_gb", "repo_root", "notes"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "gpu_name": #{ "type": "string" },
        "compute_capability": #{ "type": "string" },
        "cuda_home": #{ "type": "string" },
        "cuda_release": #{ "type": "string" },
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

let cuda_rules = "";
cuda_rules += "CUDA toolkit must be release 12.8. Never use 12.4, 13.1, a cu124 index, or whichever nvcc is first on PATH.\n";
cuda_rules += "Resolve CUDA_HOME in this order, then run `$CUDA_HOME/bin/nvcc --version` and set cuda_release to that release:\n";
cuda_rules += "1) If the supplied cuda_home argument is non-empty, use only that directory. If its nvcc is not 12.8, stop. Do not fall back.\n";
cuda_rules += "2) Else if process CUDA_HOME is set, accept it only when its nvcc release is 12.8. Ignore 12.4 or 13.1.\n";
cuda_rules += "3) Else use /usr/local/cuda-12.8 when that directory's nvcc release is 12.8.\n";
cuda_rules += "4) Do not select the newest /usr/local/cuda* entry. Do not follow a bare /usr/local/cuda symlink unless `nvcc` there is 12.8.\n";
cuda_rules += "If no 12.8 toolkit is found, ok=false. Export CUDA_HOME, PATH=$CUDA_HOME/bin:$PATH, and LD_LIBRARY_PATH from $CUDA_HOME/lib64 (or $CUDA_HOME/lib if lib64 is absent).\n";

// =============================================================================
phase("Inventory");
// =============================================================================

let inv_prompt = "";
inv_prompt += "You are inventorying MxMoE on this machine. Work in the CURRENT workspace (MxMoE repo root).\n";
inv_prompt += "READ-ONLY. Do not create, modify, or delete files. Do not conda install, pip install, build, download, or load a model.\n";
inv_prompt += "Goal: inventory hardware/toolchain so a later step can reuse or create conda env mxmoe and run kernel smoke tests.\n\n";
inv_prompt += "Run shell commands (do not guess):\n";
inv_prompt += "1) pwd; git rev-parse --show-toplevel; git branch --show-current\n";
inv_prompt += "2) nvidia-smi (note GPU name, driver, CUDA version)\n";
inv_prompt += "3) Detect compute capability (e.g. deviceQuery or python with pynvml/torch if available; else map RTX 3090→86, 4090→89, A100→80)\n";
inv_prompt += "4) which conda; conda --version; conda env list. Note whether env mxmoe exists and, if so, its python version. Do not activate a different env as a substitute.\n";
inv_prompt += "5) Resolve CUDA 12.8. Supplied cuda_home argument: '" + cuda_home + "'\n";
inv_prompt += cuda_rules;
inv_prompt += "6) Resolve a paired GCC/G++ 13: prefer gcc-13/g++-13 on PATH, otherwise use x86_64-conda-linux-gnu-gcc/g++ in the mxmoe env's bin directory. Record absolute paths and actual versions in notes.\n";
inv_prompt += "7) cmake --version; ninja --version (record; the env's copies are checked after install)\n";
inv_prompt += "8) df -h . | tail -1 (disk free)\n";
inv_prompt += "9) git submodule status | head -20; test whether mxmoe/3rdparty/fast-hadamard-transform has files beyond an empty dir\n";
inv_prompt += "10) test -f requirements.txt; test -f constraints.txt; test -f doc/environment.md; test -f 3rdparty/vllm/rust-toolchain.toml; ls perf/performance_table.json; test -f project_config.py\n";
inv_prompt += "11) If rustc is installed, record `rustc --version`. Expected channel is 1.95 from 3rdparty/vllm/rust-toolchain.toml. Do not install Rust in this phase.\n\n";
inv_prompt += "Set cuda_release to the nvcc release string, exactly 12.8 when accepted.\n";
inv_prompt += "Set ok=false if no GPU, no conda, disk free < 20GB, no GCC/G++ 13 pair can be resolved, or no CUDA 12.8 toolkit was resolved.\n";
inv_prompt += "Put blockers/warnings in notes. Do not treat a 12.4 toolkit as acceptable.\n";

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

if inv.output.cuda_release != "12.8" {
    pause("infra", "CUDA toolkit must be 12.8 (refusing 12.4 and non-12.8 auto-detect). Inventory cuda_home=" + inv.output.cuda_home);
}

log("Inventory OK: GPU=" + inv.output.gpu_name + " cc=" + inv.output.compute_capability + " cuda_home=" + inv.output.cuda_home);

if dry_run {
    let dry_report = "";
    dry_report += "# MxMoE env setup (dry_run)\n\n";
    dry_report += "Read-only. No conda create, pip install, or native build was performed.\n\n";
    dry_report += "- env_name: mxmoe (only supported env)\n";
    dry_report += "- python: 3.12 (create spec in doc/environment.md; current interpreter is 3.12.13)\n";
    dry_report += "- cuda_home: " + inv.output.cuda_home + " (release " + inv.output.cuda_release + ")\n";
    dry_report += "- skip_flash: " + skip_flash.to_string() + " (reuse a verified flash-attn build only; FlashAttention stays mandatory)\n";
    dry_report += "- execute doc/environment.md sections 2-5; do not invent a second pip line, source pin, or FA3 patch\n";
    dry_report += "- Python deps command is that guide's section 3 `python -m pip install -r requirements.txt` via conda run -n mxmoe\n";
    dry_report += "- compiled extensions stay the guide's later blocks, separate from that pip command\n";
    dry_report += "- not used: torch 2.6, cu124, a second pip index, or precompiled vLLM\n";
    dry_report += "- L2 target: conda run --no-capture-output -n mxmoe python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string() + "\n\n";
    dry_report += "## Inventory\n";
    dry_report += inv.output.notes + "\n";
    dry_report += "\nGPU: " + inv.output.gpu_name + " cc=" + inv.output.compute_capability + "\n";
    dry_report += "Would proceed to conda create/reuse + compiled extensions + L0/L1/L2 if dry_run=false.\n";
    let p = write_scratch_file("setup-mxmoe-env-dry-run.md", dry_report);
    complete(#{
        summary: "dry_run only: inventory passed for mxmoe on CUDA 12.8; no install performed",
        dry_run: true,
        env_name: "mxmoe",
        cuda_home: inv.output.cuda_home,
        report: p,
        final_test_target: "conda run --no-capture-output -n mxmoe python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string(),
    });
}

// =============================================================================
phase("Conda-Env");
// =============================================================================

let conda_prompt = "";
conda_prompt += "Create or reuse conda env mxmoe and install Python dependencies by executing doc/environment.md. Do not edit that file.\n";
conda_prompt += "Do not invent a second pip line, index URL, or requirements file. If the guide and this prompt disagree, follow the guide on commands and stop if that would install torch 2.6 or CUDA 12.4.\n";
conda_prompt += "Repo root: " + inv.output.repo_root + "\n";
conda_prompt += "CUDA_HOME already resolved: " + inv.output.cuda_home + " (nvcc must stay 12.8).\n";
conda_prompt += "force_recreate: " + force_recreate.to_string() + "\n";
conda_prompt += "skip_flash does not apply here and must not drop a dependency.\n\n";
conda_prompt += "Steps:\n";
conda_prompt += "1) Read doc/environment.md sections 2 and 3. Apply section 2 exports in this shell, using the resolved CUDA 12.8 toolkit as CUDA_HOME.\n";
conda_prompt += "2) MXMOE_BUILD_ROOT must already be set. Do not invent a machine-local path or a RAID default. If it is unset, ok=false.\n";
conda_prompt += "3) If force_recreate and env mxmoe exists: conda env remove -n mxmoe -y. Otherwise do not delete it.\n";
conda_prompt += "4) If env mxmoe is missing, run the guide's create command: conda create -n mxmoe python=3.12 -y\n";
conda_prompt += "5) If env mxmoe exists, reuse it. The current interpreter is Python 3.12.13. Do not replace it because the create spec says 3.12. Refuse a non-3.12 interpreter unless force_recreate already replaced the env.\n";
conda_prompt += "6) Activate mxmoe or use conda run --no-capture-output -n mxmoe. sys.prefix must end with /envs/mxmoe. Never use base, bitmoe, or system Python.\n";
conda_prompt += "7) Before pip, initialize submodules from the repo root. requirements.txt includes 3rdparty/vllm/requirements/cuda.txt and requirements/build/cuda.txt; a fresh checkout fails if that submodule is still empty.\n";
conda_prompt += "   git submodule update --init --recursive\n";
conda_prompt += "   Do not pass --force and do not reset. Local submodule fixes must stay.\n";
conda_prompt += "   Stop with ok=false if 3rdparty/vllm/requirements/cuda.txt or requirements/build/cuda.txt is missing after that command.\n";
conda_prompt += "8) Read requirements.txt. If it still pins torch==2.6.0, flash_attn==2.7.4.post1, or nvidia cu12 wheels, ok=false. Do not install that file.\n";
conda_prompt += "9) Otherwise run the guide's section 3 Python command exactly, and do not append another --extra-index-url or -c (requirements.txt already carries both):\n";
conda_prompt += "   conda run --no-capture-output -n mxmoe python -m pip install -r requirements.txt\n";
conda_prompt += "   pycuda may compile as part of that file. That is still this phase. Do not abort unless the install fails.\n";
conda_prompt += "   Retry once on a transient network error. If installed distributions already satisfy constraints.txt and the guide's version table, do not reinstall them.\n";
conda_prompt += "10) This phase stops before the guide's Hadamard, FlashAttention, and vLLM blocks. Do not compile those here.\n";
conda_prompt += "11) Do not set VLLM_USE_PRECOMPILED or any VLLM_USE_PRECOMPILED* variable. Do not download HF models.\n";
conda_prompt += "12) Run the guide's section 3 version print. Require torch 2.11.0+cu128, torch.version.cuda 12.8, transformers 5.19.0. Reject torch 2.6 and transformers 4.x.\n";
conda_prompt += "Return ok=true only if that print matches and torch.cuda.is_available() is True inside mxmoe.\n";
conda_prompt += "Put the env prefix in env_path. Include the last 40 lines of install log in log_tail if long.\n";

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
prep_prompt += "Prepare the MxMoE repo and compiled extensions on THIS node. Repo: " + inv.output.repo_root + "\n";
prep_prompt += "Use conda env mxmoe only. source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate mxmoe\n";
prep_prompt += "Every pip command is `python -m pip` from that interpreter. Do not use bare pip, base, or another env.\n";
prep_prompt += "Detected compute capability for tile tables: " + inv.output.compute_capability + "\n";
prep_prompt += "CUDA_HOME: " + inv.output.cuda_home + " (must remain 12.8)\n";
prep_prompt += "skip_flash: " + skip_flash.to_string() + " — this may skip a FlashAttention rebuild only when the checks below already pass. It must not skip the package.\n";
prep_prompt += "Host compiler for every native build: resolve CC/CXX as the GCC/G++ 13 pair from inventory (system gcc-13/g++-13 or the mxmoe Conda compiler pair). Do not fall back to another version.\n";
prep_prompt += "Do not reinstall a package that is already verified compatible with this torch and this GPU.\n";
prep_prompt += "Do not download HF models. Do not edit doc/environment.md or write under docs/.\n\n";
prep_prompt += cuda_rules;
prep_prompt += "\nExecute ALL of the following:\n\n";
prep_prompt += "### A) Submodules\n";
prep_prompt += "git submodule update --init --recursive was required before the requirements install. Re-run it here; it is idempotent. Do not pass --force and do not reset.\n";
prep_prompt += "Confirm 3rdparty/vllm/requirements/{cuda.txt,build/cuda.txt} exist, mxmoe/3rdparty/fast-hadamard-transform has setup.py, and mxmoe/kernels/src/3rdparty/{cutlass,cccl,fmt,pybind11} are non-empty.\n\n";
prep_prompt += "### B) Compiled extensions — execute doc/environment.md, do not restate it\n";
prep_prompt += "Read sections 3 and 4 and run those blocks. Do not copy source-cache pins, the FA3 patch, or MXMOE_BUILD_ROOT paths into another recipe.\n";
prep_prompt += "If MXMOE_BUILD_ROOT is unset, stop. Do not invent a directory.\n";
prep_prompt += "Apply the guide's section 2 exports first (CUDA 12.8, GCC/G++ 13, the guide's MAX_JOBS / NVCC_THREADS, unset VLLM_USE_PRECOMPILED*).\n";
prep_prompt += "Reuse a Hadamard or standalone flash-attn build only when the guide's same-version numeric smoke already passes against this torch and this GPU. Otherwise run the guide's install block, even if skip_flash is true.\n";
prep_prompt += "skip_flash never skips vLLM's vllm_flash_attn and never waives either package.\n";
prep_prompt += "Do not reset local submodule fixes. The Hadamard CUDA 13 sm_70 guard described in the guide must stay.\n";
prep_prompt += "For SM80, keep the guide's TORCH_CUDA_ARCH_LIST and FLASH_ATTN_CUDA_ARCHS. For another GPU, reselect both for that GPU. Do not treat an SM80 build as compatible with a different arch. Do not add 9.x unless the GPU is 9.x.\n";
prep_prompt += "vLLM must be the guide's editable install of 3rdparty/vllm, never a precompiled wheel. A version prefix 0.32.1.dev0 is the v0.32.0 baseline, not a different release.\n";
prep_prompt += "Do not call LLM.generate or load a model. Import success is not a generate pass.\n\n";
prep_prompt += "### C) GPU arch patches (required when this GPU's cc is missing from the tile tables)\n";
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
prep_prompt += "### F) Model paths live in the gitignored repo-root .env (MXMOE_MODEL_* keys; see .env.example). Do NOT edit tracked project_config.py; L2 should not load HF weights, so .env model keys are not required here.\n\n";
prep_prompt += "Return ok=true only if submodules are present, hadamard 1.0.4.post1 imports, flash-attn 2.8.3.post2 imports, vLLM is the editable checkout, the arch patch is applied (or already correct for this cc), and the gate json exists.\n";

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
v01_prompt += "Run L0 and L1 by executing doc/environment.md section 5. Do not rewrite its scripts or tolerances.\n";
v01_prompt += "Repo: " + inv.output.repo_root + "\n";
v01_prompt += "CUDA_HOME: " + inv.output.cuda_home + " (nvcc release must be 12.8, not 12.4 or 13.1).\n";
v01_prompt += "skip_flash=" + skip_flash.to_string() + " does not waive FlashAttention, Hadamard, or vLLM.\n";
v01_prompt += "Use conda run --no-capture-output -n mxmoe. Do not use another env.\n\n";
v01_prompt += "### L0 — guide import check, plus project imports\n";
v01_prompt += "Run the section 5 block headed 依赖及源码来源. It must print ENV_IMPORT_PASS.\n";
v01_prompt += "Also import datasets, einops, regex, jsbeautifier, pynvml, pycuda.driver, gurobipy (import only), tile_config.get_gpu_info, and compose_kernel.TemplateGenerator.\n";
v01_prompt += "Python must be 3.12 (the existing interpreter is 3.12.13) and sys.prefix must end with /envs/mxmoe.\n";
v01_prompt += "Do not call vllm.LLM or LLM.generate and do not load a model. Print L0 PASS only after ENV_IMPORT_PASS.\n\n";
v01_prompt += "### L1 — guide numeric smoke\n";
v01_prompt += "Run the section 5 block headed 小规模 CUDA 数值 smoke. Set MXMOE_TEST_GPU in the shell to a free GPU; do not occupy someone else's job and do not write that id into a tracked file.\n";
v01_prompt += "Require the line CUDA_NUMERIC_PASS. That smoke is not a performance result and not an LLM.generate pass. Print L1 PASS.\n\n";
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
l2_prompt += "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate mxmoe\n";
l2_prompt += "The python on PATH must be the mxmoe interpreter. Do not use another env.\n";
l2_prompt += "export CUDA_HOME=" + inv.output.cuda_home + "\n";
l2_prompt += "export PATH=$CUDA_HOME/bin:$PATH\n";
l2_prompt += "Resolve the GCC/G++ 13 pair from Inventory again and export its absolute paths as CC/CXX (system or mxmoe Conda pair).\n";
l2_prompt += "Use this env's cmake and ninja. Do not apt-install a second toolchain. The verified env has cmake 4.4.4 and ninja 1.11.1.4; do not fail solely on a newer build that still drives this Ninja build.\n";
l2_prompt += "Confirm nvcc is 12.8 before building.\n\n";
l2_prompt += "### Final command (must run from repo root)\n";
l2_prompt += "python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string() + "\n";
l2_prompt += "Equivalent explicit form: conda run --no-capture-output -n mxmoe python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string() + "\n";
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

let final_cmd = "conda run --no-capture-output -n mxmoe python run_mxmoe_gg.py --model qwen2_moe --dataset wiki2 --bs " + batch_size.to_string() + " --layer " + layer.to_string();

let report_prompt = "";
report_prompt += "Write a concise markdown setup report for conda env mxmoe on this node.\n";
report_prompt += "Use facts from the completed steps (do not invent success):\n\n";
report_prompt += "Inventory notes: " + inv.output.notes + "\n";
report_prompt += "GPU: " + inv.output.gpu_name + " cc=" + inv.output.compute_capability + "\n";
report_prompt += "CUDA_HOME: " + inv.output.cuda_home + " release=" + inv.output.cuda_release + "\n";
report_prompt += "Conda summary: " + conda_r.output.summary + "\n";
report_prompt += "Conda details: " + conda_r.output.details + "\n";
report_prompt += "Repo prep: " + prep_r.output.summary + "\n";
report_prompt += "L0/L1: " + v01.output.summary + "\n";
report_prompt += "L2 summary: " + l2.output.summary + "\n";
report_prompt += "L2 bench_path: " + l2.output.bench_path + "\n";
report_prompt += "skip_flash: " + skip_flash.to_string() + " (reuse only; FlashAttention was still required)\n\n";
report_prompt += "Report MUST include sections:\n";
report_prompt += "1. Environment (mxmoe, Python 3.12, torch 2.11.0+cu128, CUDA 12.8, doc/environment.md)\n";
report_prompt += "2. Compiled extensions as verified by the guide's import check and CUDA_NUMERIC_PASS — not an LLM.generate pass\n";
report_prompt += "3. Patches applied (CMake arch, tile_config alias, synthetic trace path)\n";
report_prompt += "4. Verification results (L0/L1/L2)\n";
report_prompt += "5. Final test target command that passed: " + final_cmd + "\n";
report_prompt += "6. How to activate: conda activate mxmoe\n";
report_prompt += "7. Known non-goals (no Gurobi license, no HF model download, no bits_solver, no vLLM LLM.generate, no torch 2.6/cu124)\n";
report_prompt += "8. Next steps for real experiments (set MXMOE_MODEL_* in gitignored .env, optional Gurobi license). lm_eval==0.4.12 comes from requirements.txt; install-flash-lmeval only re-verifies FA2 and lm_eval.\n\n";
report_prompt += "Do NOT write under repo docs/ or edit doc/environment.md.\n";
report_prompt += "Return the full markdown report text in details, and ok=true.\n";

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

let scratch = "";
scratch += "# MxMoE setup-mxmoe-env result\n\n";
scratch += "**Final test target (PASSED):** `" + final_cmd + "`\n\n";
scratch += "- env: mxmoe\n";
scratch += "- CUDA_HOME: " + inv.output.cuda_home + "\n";
scratch += "- GPU: " + inv.output.gpu_name + " (cc " + inv.output.compute_capability + ")\n";
scratch += "- L2 bench: " + l2.output.bench_path + "\n";
scratch += "- report: " + report_path + "\n";
let sp = write_scratch_file("setup-mxmoe-env-result.md", scratch);

complete(#{
    summary: "conda env mxmoe ready on CUDA 12.8; L0/L1/L2 passed",
    env_name: "mxmoe",
    cuda_home: inv.output.cuda_home,
    gpu: inv.output.gpu_name,
    compute_capability: inv.output.compute_capability,
    final_test_target: final_cmd,
    final_test_meaning: "FP16 GroupGEMM path: synthetic/real gate trace → template kernel codegen → CMake/NVCC build → ./build/test bench. Proves Python deps, CUDA 12.8 toolchain, submodules, and arch patches without downloading MoE weights.",
    bench_path: l2.output.bench_path,
    report_path: report_path,
    scratch: sp,
    l2_ok: true,
});
