// install-flash-lmeval: Reuse or compile-install standalone flash-attn into
// conda env mxmoe, and confirm lm_eval from the shared requirements pin.
// Does not create another env, reinstall torch, or install precompiled vLLM.
// FlashAttention executes doc/environment.md section 3. lm_eval==0.4.12 is not
// a second pip spec. CUDA toolkit is 12.8.
//
// Skip flags reuse an already verified compatible package. They do not omit a
// mandatory dependency and cannot make Verify pass while a package is missing.
//
// Args (all optional):
//   env_name          default "mxmoe" (any other value is rejected)
//   cuda_home         default "" (this arg if it is CUDA 12.8, else discover
//                     a 12.8 toolkit; never /usr/local/cuda-12.4 or newest cuda*)
//   flash_version     default "2.8.3.post2" (any other value is rejected)
//   lm_eval_spec      default "lm_eval==0.4.12" (any other value is rejected)
//   max_jobs          default "2" (matches doc/environment.md section 2)
//   torch_arch_list   default "8.0"   (A100 sm80; do not add 9.x unless the GPU is 9.x)
//   gpu_devices       default "2,3"
//   force_reinstall   default false
//   skip_flash        default false (reuse only when flash-attn 2.8.3.post2 already passes CUDA smoke)
//   skip_lm_eval      default false (reuse only when lm_eval 0.4.12 already imports)
//   repo_root         default "" (agent uses workspace / git toplevel)
//

let meta = #{
    name: "install-flash-lmeval",
    description: "Reuse or compile-install flash-attn 2.8.3.post2 and confirm lm_eval 0.4.12 in conda env mxmoe on CUDA 12.8, then verify FA2 and the project Evaluator",
    when_to_use: "When flash_attention_2 and lm_eval must be present in the existing mxmoe performance stack",
    phases: [
        #{ title: "Precheck", detail: "mxmoe env, CUDA 12.8, existing torch cu128, disk" },
        #{ title: "Install-lm_eval", detail: "reuse lm_eval==0.4.12 from requirements.txt; do not install another spec" },
        #{ title: "Install-flash_attn", detail: "reuse or execute the guide's section 3 flash-attn block" },
        #{ title: "Verify", detail: "guide numeric smoke plus lm_eval, Evaluator, and the transformers FA2 flag" },
        #{ title: "Report", detail: "write install report artifact" },
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

let env_name = arg_str(args, "env_name", "mxmoe");
let cuda_home = arg_str(args, "cuda_home", "");
let flash_version = arg_str(args, "flash_version", "2.8.3.post2");
let lm_eval_spec = arg_str(args, "lm_eval_spec", "lm_eval==0.4.12");
let max_jobs = arg_str(args, "max_jobs", "2");
let torch_arch_list = arg_str(args, "torch_arch_list", "8.0");
let gpu_devices = arg_str(args, "gpu_devices", "2,3");
let force_reinstall = arg_bool(args, "force_reinstall", false);
let skip_flash = arg_bool(args, "skip_flash", false);
let skip_lm_eval = arg_bool(args, "skip_lm_eval", false);
let repo_root_arg = arg_str(args, "repo_root", "");

if env_name != "mxmoe" {
    pause("infra", "Refusing env_name='" + env_name + "'. The only supported conda environment is mxmoe.");
}
if flash_version != "2.8.3.post2" {
    pause("infra", "Refusing flash_version='" + flash_version + "'. The mxmoe stack uses flash-attn 2.8.3.post2.");
}
if lm_eval_spec != "lm_eval==0.4.12" {
    pause("infra", "Refusing lm_eval_spec='" + lm_eval_spec + "'. The mxmoe stack uses lm_eval==0.4.12.");
}

let step_schema = #{
    "type": "object",
    "required": ["ok", "summary", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "details": #{ "type": "string" },
        "env_path": #{ "type": "string" },
        "version": #{ "type": "string" },
        "cuda_release": #{ "type": "string" },
        "log_tail": #{ "type": "string" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["ok", "summary", "flash_ok", "lm_eval_ok", "project_ok", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "flash_ok": #{ "type": "boolean" },
        "lm_eval_ok": #{ "type": "boolean" },
        "project_ok": #{ "type": "boolean" },
        "flash_version": #{ "type": "string" },
        "lm_eval_version": #{ "type": "string" },
        "details": #{ "type": "string" },
    },
};

let cuda_rules = "";
cuda_rules += "CUDA toolkit must be release 12.8. Never use 12.4, 13.1, or whichever nvcc is first on PATH.\n";
cuda_rules += "Supplied cuda_home argument: '" + cuda_home + "'.\n";
cuda_rules += "If that argument is non-empty, use only that directory and require nvcc release 12.8. Do not fall back.\n";
cuda_rules += "If it is empty, do not export an empty CUDA_HOME and do not keep an inherited 12.4 or 13.1 CUDA_HOME.\n";
cuda_rules += "Accept process CUDA_HOME only when its nvcc is 12.8; otherwise use /usr/local/cuda-12.8 when that nvcc is 12.8.\n";
cuda_rules += "Do not select the newest /usr/local/cuda* entry. Do not follow a bare /usr/local/cuda symlink unless nvcc is 12.8.\n";
cuda_rules += "Export CUDA_HOME, PATH=$CUDA_HOME/bin:$PATH, and LD_LIBRARY_PATH from $CUDA_HOME/lib64 (or lib if lib64 is absent).\n";

// =============================================================================
phase("Precheck");
// =============================================================================

let pre_prompt = "";
pre_prompt += "READ-ONLY precheck for flash-attn and lm_eval in the existing mxmoe performance stack.\n";
pre_prompt += "Do not pip install, conda create, build, or load a model.\n";
pre_prompt += "Do not use base, bitmoe, or any env other than mxmoe.\n\n";
pre_prompt += "Config:\n";
pre_prompt += "- env_name: mxmoe\n";
pre_prompt += "- gpu CUDA_VISIBLE_DEVICES: " + gpu_devices + "\n";
pre_prompt += "- TORCH_CUDA_ARCH_LIST: " + torch_arch_list + "\n";
pre_prompt += "- repo_root hint: " + repo_root_arg + "\n\n";
pre_prompt += cuda_rules;
pre_prompt += "\nExecute (shell, do not guess):\n";
pre_prompt += "1) source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda env list | grep mxmoe\n";
pre_prompt += "2) ENV_PY=$(conda run --no-capture-output -n mxmoe which python)\n";
pre_prompt += "3) conda run --no-capture-output -n mxmoe python -c \"import sys,torch; print(sys.prefix); print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"\n";
pre_prompt += "4) Resolve CUDA 12.8 as above; `$CUDA_HOME/bin/nvcc --version`\n";
pre_prompt += "5) df -h . | tail -1\n";
pre_prompt += "6) pwd; git rev-parse --show-toplevel 2>/dev/null || true; test -f doc/environment.md; test -f constraints.txt\n";
pre_prompt += "7) conda run --no-capture-output -n mxmoe python -c \"import flash_attn,lm_eval; print(flash_attn.__version__, lm_eval.__version__)\" (report missing imports; do not install)\n\n";
pre_prompt += "ok=true only if env mxmoe exists, Python is 3.12 (current interpreter 3.12.13), torch.__version__ is 2.11.0+cu128, torch.version.cuda is 12.8, nvcc release is 12.8, and disk free > 10GB.\n";
pre_prompt += "If torch is 2.6 or CUDA is not 12.8, ok=false. Do not upgrade torch here; point at setup-mxmoe-env and doc/environment.md.\n";
pre_prompt += "Set cuda_release to 12.8 when accepted. Put the env prefix in env_path. Put notes in details.\n";

let pre = agent(pre_prompt, #{
    label: "precheck",
    capability_mode: "all",
    output_schema: step_schema,
});

if pre == () || !pre.success || pre.output == () {
    pause("infra", "Precheck agent failed. Resume after fixing shell/conda access.");
}
if pre.output.ok != true {
    pause("infra", "Precheck not-ok: " + pre.output.summary + " | " + pre.output.details);
}
if pre.output.cuda_release != "12.8" {
    pause("infra", "Precheck CUDA is not 12.8. Refusing 12.4 or an unresolved toolkit.");
}
log("Precheck OK: " + pre.output.summary);

// =============================================================================
phase("Install-lm_eval");
// =============================================================================

let lm_prompt = "";
lm_prompt += "Confirm lm_eval==0.4.12 in conda env mxmoe. It is a requirements.txt pin, not a compiled extension.\n";
lm_prompt += "skip_lm_eval: " + skip_lm_eval.to_string() + "\n";
lm_prompt += "force_reinstall: " + force_reinstall.to_string() + "\n";
lm_prompt += "Do not pip install a different spec. Do not pass lm_eval[hf].\n";
lm_prompt += "env_path hint: " + pre.output.env_path + "\n\n";
lm_prompt += "Steps:\n";
lm_prompt += "1) conda run --no-capture-output -n mxmoe python -c \"import sys; print(sys.prefix)\"  # must end with /envs/mxmoe\n";
lm_prompt += "2) If import lm_eval works and lm_eval.__version__ == '0.4.12' and force_reinstall is false: reuse it. skip_lm_eval does not change that.\n";
lm_prompt += "3) If it is missing or not 0.4.12, install via the guide's section 3 command, even if skip_lm_eval is true.\n";
lm_prompt += "   First, from the repo root: git submodule update --init --recursive\n";
lm_prompt += "   Do not pass --force or reset. requirements.txt includes 3rdparty/vllm requirement manifests; stop if those files are still missing.\n";
lm_prompt += "   conda run --no-capture-output -n mxmoe python -m pip install -r requirements.txt\n";
lm_prompt += "   The requirements file already carries its index and constraints; use the targeted reinstall below only for an explicit force_reinstall request.\n";
lm_prompt += "4) If force_reinstall is true and the version is already 0.4.12, reinstall only lm_eval with the same constraint file: conda run --no-capture-output -n mxmoe python -m pip install --force-reinstall --no-deps -c constraints.txt lm_eval==0.4.12. Do not reinstall torch.\n";
lm_prompt += "5) conda run --no-capture-output -n mxmoe python -c \"import lm_eval; print(lm_eval.__version__)\" must print 0.4.12.\n";
lm_prompt += "Do not reinstall torch. Do not use bare pip.\n\n";
lm_prompt += "ok=true only if import lm_eval succeeds and the version is 0.4.12.\n";
lm_prompt += "Put the version string in version. Put the pip log tail in log_tail if an install ran.\n";

let lm_r = agent(lm_prompt, #{
    label: "install-lm-eval",
    capability_mode: "all",
    output_schema: step_schema,
});

if lm_r == () || !lm_r.success || lm_r.output == () || lm_r.output.ok != true {
    let d = "lm_eval install failed";
    if lm_r != () && lm_r.success && lm_r.output != () {
        d = lm_r.output.summary + " | " + lm_r.output.details;
    }
    pause("infra", "lm_eval install failed: " + d);
}
log("lm_eval ready: " + lm_r.output.summary);

// =============================================================================
phase("Install-flash_attn");
// =============================================================================

let fa_prompt = "";
fa_prompt += "Reuse or compile standalone flash-attn by executing doc/environment.md section 3. Do not restate a second build recipe.\n";
fa_prompt += "This package is mandatory. skip_flash=" + skip_flash.to_string() + " may skip the compile only when version 2.8.3.post2 already passes the guide's numeric smoke on this GPU and this torch.\n";
fa_prompt += "The section 4 FA3 source-cache patch belongs to the vLLM editable build, not this standalone package. Do not copy that patch here.\n";
fa_prompt += "Do not edit doc/environment.md. If MXMOE_BUILD_ROOT is unset, stop rather than inventing a path.\n\n";
fa_prompt += "Apply the guide's section 2 exports first, with this resolved CUDA 12.8 toolkit:\n";
fa_prompt += cuda_rules;
fa_prompt += "Then set MAX_JOBS=" + max_jobs + " (guide default is 2). Requested TORCH_CUDA_ARCH_LIST=\"" + torch_arch_list + "\".\n";
fa_prompt += "If this GPU is not SM80, reselect TORCH_CUDA_ARCH_LIST and FLASH_ATTN_CUDA_ARCHS for that GPU as the guide requires. Do not add 9.x unless the GPU is 9.x.\n";
fa_prompt += "export CUDA_VISIBLE_DEVICES=" + gpu_devices + "\n";
fa_prompt += "force_reinstall: " + force_reinstall.to_string() + "\n\n";
fa_prompt += "Steps:\n";
fa_prompt += "1) Confirm mxmoe python and torch 2.11.0+cu128. Do not reinstall torch.\n";
fa_prompt += "2) If force_reinstall is false and flash-attn 2.8.3.post2 already passes the guide's section 5 numeric smoke, reuse it regardless of skip_flash.\n";
fa_prompt += "3) Otherwise run the section 3 flash-attn install block, even if skip_flash is true. Add --force-reinstall when repairing an installed incompatible build or when force_reinstall is true. Keep --no-cache-dir, --no-binary, -c constraints.txt, and --no-deps: a cached wheel can retain the old Torch/CUDA ABI even when FORCE_BUILD is set.\n";
fa_prompt += "4) conda run --no-capture-output -n mxmoe python -c \"import flash_attn; print(flash_attn.__version__)\" must print 2.8.3.post2.\n\n";
fa_prompt += "Do not use bare pip. Do not set VLLM_USE_PRECOMPILED.\n";
fa_prompt += "ok=true only if import flash_attn succeeds at 2.8.3.post2.\n";
fa_prompt += "Put the version in version. Include the last ~40 lines of build log in log_tail.\n";

let fa_r = agent(fa_prompt, #{
    label: "install-flash-attn",
    capability_mode: "all",
    output_schema: step_schema,
});

if fa_r == () || !fa_r.success || fa_r.output == () || fa_r.output.ok != true {
    let d = "flash_attn install failed";
    if fa_r != () && fa_r.success && fa_r.output != () {
        d = fa_r.output.summary + " | " + fa_r.output.details;
    }
    pause("infra", "flash_attn compile/install failed: " + d);
}
log("flash_attn ready: " + fa_r.output.summary);

// =============================================================================
phase("Verify");
// =============================================================================

let v_prompt = "";
v_prompt += "FINAL verification: flash_attn and lm_eval are both required. Skip flags do not waive either one.\n";
v_prompt += "Use conda run --no-capture-output -n mxmoe. Repo root is git rev-parse --show-toplevel.\n";
v_prompt += "export CUDA_VISIBLE_DEVICES=" + gpu_devices + "\n";
v_prompt += cuda_rules;
v_prompt += "Do not load a HF model. Do not call vLLM LLM.generate.\n\n";
v_prompt += "### A) Run doc/environment.md section 5 numeric smoke (required)\n";
v_prompt += "Do not rewrite its tensors or tolerances. Set MXMOE_TEST_GPU for this shell to a free GPU (first id in " + gpu_devices + " if MXMOE_TEST_GPU is unset). Do not write that id into a tracked file.\n";
v_prompt += "Require CUDA_NUMERIC_PASS. Also assert is_flash_attn_2_available() and flash_attn.__version__ == '2.8.3.post2'.\n";
v_prompt += "Print FLASH_ATTN PASS\n\n";
v_prompt += "### B) lm_eval + project Evaluator (required)\n";
v_prompt += "import lm_eval; assert lm_eval.__version__ == '0.4.12'\n";
v_prompt += "from lm_eval import tasks\n";
v_prompt += "from lm_eval.models.huggingface import HFLM\n";
v_prompt += "from mxmoe.quant.evaluator import Evaluator\n";
v_prompt += "Try tasks.get_task_dict(['arc_easy','piqa','hellaswag','winogrande'])\n";
v_prompt += "  if API fails, use TaskManager and match_tasks — still ok if tasks resolve\n";
v_prompt += "Print LM_EVAL PASS and PROJECT PASS\n\n";
v_prompt += "### C) Project usage note\n";
v_prompt += "Grep confirms quant.py / moe_tracer / gptq use attn_implementation=flash_attention_2 for non-remote models, and evaluator.eval_tasks uses lm_eval. Report that in details.\n";
v_prompt += "Transformers 5 uses eager attention for remote-code models; that does not waive flash-attn for this stack.\n\n";
v_prompt += "Set flash_ok, lm_eval_ok, and project_ok. ok=true only if all three pass.\n";
v_prompt += "A missing package is a failure even if skip_flash or skip_lm_eval was set.\n";

let ver = agent(v_prompt, #{
    label: "verify-both",
    capability_mode: "all",
    output_schema: verify_schema,
});

if ver == () || !ver.success || ver.output == () {
    pause("verification", "Verify agent failed to return structured result.");
}
if ver.output.ok != true || ver.output.flash_ok != true || ver.output.lm_eval_ok != true || ver.output.project_ok != true {
    pause("verification", "Verify FAILED: " + ver.output.summary + " | " + ver.output.details);
}
log("Verify PASS: " + ver.output.summary);

// =============================================================================
phase("Report");
// =============================================================================

let rep_prompt = "";
rep_prompt += "Write a short markdown report of the flash-attn + lm_eval install for conda env mxmoe.\n";
rep_prompt += "cuda_home resolved during precheck; release must be 12.8.\n";
rep_prompt += "gpu: " + gpu_devices + "\n";
rep_prompt += "precheck: " + pre.output.summary + "\n";
rep_prompt += "lm_eval install: " + lm_r.output.summary + " ver=" + lm_r.output.version + "\n";
rep_prompt += "flash_attn install: " + fa_r.output.summary + " ver=" + fa_r.output.version + "\n";
rep_prompt += "verify: " + ver.output.summary + "\n";
rep_prompt += "flash_ok=" + ver.output.flash_ok.to_string() + " lm_eval_ok=" + ver.output.lm_eval_ok.to_string() + "\n";
rep_prompt += "skip_flash=" + skip_flash.to_string() + " skip_lm_eval=" + skip_lm_eval.to_string() + " (reuse only; both packages were still required)\n\n";
rep_prompt += "Include: conda activate mxmoe, flash-attn 2.8.3.post2, lm_eval 0.4.12, pointer to doc/environment.md, note that quant/eval use FA2 + lm_eval.\n";
rep_prompt += "Do not claim a vLLM generate pass. Do NOT write under repo docs/ or edit doc/environment.md.\n";
rep_prompt += "Return the full markdown report text in details, and ok=true.\n";

let rep = agent(rep_prompt, #{
    label: "write-report",
    capability_mode: "all",
    output_schema: step_schema,
});

let report_body = "";
if rep != () && rep.success && rep.output != () && rep.output.details != () {
    report_body = rep.output.details;
}
let report_path = write_scratch_file("install-flash-lmeval-report.md", report_body);

let scratch = "";
scratch += "# install-flash-lmeval result\n\n";
scratch += "- env: mxmoe\n";
scratch += "- flash_ok: " + ver.output.flash_ok.to_string() + "\n";
scratch += "- lm_eval_ok: " + ver.output.lm_eval_ok.to_string() + "\n";
scratch += "- project_ok: " + ver.output.project_ok.to_string() + "\n";
scratch += "- report: " + report_path + "\n";
let sp = write_scratch_file("install-flash-lmeval-result.md", scratch);

complete(#{
    summary: "flash-attn 2.8.3.post2 and lm_eval 0.4.12 verified in conda env mxmoe",
    env_name: "mxmoe",
    flash_ok: ver.output.flash_ok,
    lm_eval_ok: ver.output.lm_eval_ok,
    project_ok: ver.output.project_ok,
    flash_version: ver.output.flash_version,
    lm_eval_version: ver.output.lm_eval_version,
    report_path: report_path,
    scratch: sp,
});
