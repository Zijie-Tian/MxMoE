// install-flash-lmeval: Compile-install flash_attn + lm_eval into the isolated
// mxmoe conda env, then verify both work for this MxMoE repo (GPU path).
//
// Args (all optional):
//   env_name          default "mxmoe"
//   cuda_home         default "/usr/local/cuda-12.4"
//   flash_version     default "2.7.4.post1"  (matches requirements.txt)
//   lm_eval_spec      default "lm_eval>=0.4.0"
//   max_jobs          default "8"
//   torch_arch_list   default "8.0"   (A100 sm80; set "8.0;8.9" if multi-arch)
//   gpu_devices       default "2,3"
//   force_reinstall   default false
//   skip_flash        default false
//   skip_lm_eval      default false
//   repo_root         default "" (agent uses workspace / git toplevel)
//
let meta = #{
    name: "install-flash-lmeval",
    description: "Compile-install flash_attn and install lm_eval into conda env mxmoe, then smoke-test both for MxMoE",
    when_to_use: "When you need flash_attention_2 and lm_eval zero-shot eval ready in the mxmoe env on this machine",
    phases: [
        #{ title: "Precheck", detail: "conda env, CUDA, torch, disk" },
        #{ title: "Install-lm_eval", detail: "pip install lm_eval (wheels/source)" },
        #{ title: "Install-flash_attn", detail: "pip compile-install flash-attn --no-build-isolation" },
        #{ title: "Verify", detail: "CUDA flash_attn_func + project Evaluator/lm_eval + transformers FA2 flag" },
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
let cuda_home = arg_str(args, "cuda_home", "/usr/local/cuda-12.4");
let flash_version = arg_str(args, "flash_version", "2.7.4.post1");
let lm_eval_spec = arg_str(args, "lm_eval_spec", "lm_eval>=0.4.0");
let max_jobs = arg_str(args, "max_jobs", "8");
let torch_arch_list = arg_str(args, "torch_arch_list", "8.0");
let gpu_devices = arg_str(args, "gpu_devices", "2,3");
let force_reinstall = arg_bool(args, "force_reinstall", false);
let skip_flash = arg_bool(args, "skip_flash", false);
let skip_lm_eval = arg_bool(args, "skip_lm_eval", false);
let repo_root_arg = arg_str(args, "repo_root", "");

let step_schema = #{
    "type": "object",
    "required": ["ok", "summary", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "details": #{ "type": "string" },
        "env_path": #{ "type": "string" },
        "version": #{ "type": "string" },
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

// =============================================================================
phase("Precheck");
// =============================================================================

let pre_prompt = "";
pre_prompt += "You are preparing the MxMoE conda env for flash_attn (compile) and lm_eval.\n";
pre_prompt += "Work on this machine. Prefer explicit paths under the mxmoe env — do NOT use base or kitty.\n\n";
pre_prompt += "Config:\n";
pre_prompt += "- env_name: " + env_name + "\n";
pre_prompt += "- cuda_home: " + cuda_home + "\n";
pre_prompt += "- gpu CUDA_VISIBLE_DEVICES: " + gpu_devices + "\n";
pre_prompt += "- TORCH_CUDA_ARCH_LIST: " + torch_arch_list + "\n";
pre_prompt += "- repo_root hint: " + repo_root_arg + "\n\n";
pre_prompt += "Execute (shell, do not guess):\n";
pre_prompt += "1) source conda.sh; conda env list | grep " + env_name + "\n";
pre_prompt += "2) ENV_PY=$(conda run -n " + env_name + " which python) or /path/to/envs/" + env_name + "/bin/python\n";
pre_prompt += "3) $ENV_PY -c \"import sys,torch; print(sys.prefix); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"\n";
pre_prompt += "4) export CUDA_HOME=" + cuda_home + "; $CUDA_HOME/bin/nvcc --version | tail -2\n";
pre_prompt += "5) df -h . | tail -1\n";
pre_prompt += "6) pwd; git rev-parse --show-toplevel 2>/dev/null || true\n";
pre_prompt += "7) Check current: $ENV_PY -c \"import flash_attn\" and import lm_eval (report installed or missing)\n\n";
pre_prompt += "ok=true only if env exists, torch imports, CUDA_HOME/nvcc present, disk free > 10GB.\n";
pre_prompt += "Put env prefix in env_path. Put notes in details.\n";

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
log("Precheck OK: " + pre.output.summary);

// =============================================================================
phase("Install-lm_eval");
// =============================================================================

let lm_r = ();
if skip_lm_eval {
    log("skip_lm_eval=true — skipping lm_eval install");
} else {
    let lm_prompt = "";
    lm_prompt += "Install lm_eval into ISOLATED conda env `" + env_name + "` only.\n";
    lm_prompt += "force_reinstall: " + force_reinstall.to_string() + "\n";
    lm_prompt += "pip package spec: " + lm_eval_spec + "\n";
    lm_prompt += "env_path hint: " + pre.output.env_path + "\n\n";
    lm_prompt += "Steps:\n";
    lm_prompt += "1) source conda.sh; conda activate " + env_name + "\n";
    lm_prompt += "2) which python; python -c \"import sys; print(sys.prefix)\"  # must be .../envs/" + env_name + "\n";
    lm_prompt += "3) If already importable and force_reinstall=false: print version and skip pip\n";
    lm_prompt += "4) Else: pip install '" + lm_eval_spec + "'  (retry once on network error)\n";
    lm_prompt += "5) python -c \"import lm_eval; print(lm_eval.__version__)\"\n\n";
    lm_prompt += "ok=true only if import lm_eval succeeds in that env.\n";
    lm_prompt += "Put version string in version. Put pip log tail in log_tail if long.\n";

    lm_r = agent(lm_prompt, #{
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
}

// =============================================================================
phase("Install-flash_attn");
// =============================================================================

let fa_r = ();
if skip_flash {
    log("skip_flash=true — skipping flash_attn compile install");
} else {
    let fa_prompt = "";
    fa_prompt += "COMPILE-INSTALL flash-attn into ISOLATED conda env `" + env_name + "`.\n";
    fa_prompt += "This is a long CUDA compile. Do not skip. Use --no-build-isolation.\n\n";
    fa_prompt += "Env:\n";
    fa_prompt += "  export CUDA_VISIBLE_DEVICES=" + gpu_devices + "\n";
    fa_prompt += "  export CUDA_HOME=" + cuda_home + "\n";
    fa_prompt += "  export PATH=$CUDA_HOME/bin:$PATH\n";
    fa_prompt += "  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n";
    fa_prompt += "  export MAX_JOBS=" + max_jobs + "\n";
    fa_prompt += "  export TORCH_CUDA_ARCH_LIST=\"" + torch_arch_list + "\"\n\n";
    fa_prompt += "force_reinstall: " + force_reinstall.to_string() + "\n";
    fa_prompt += "version pin: " + flash_version + "\n\n";
    fa_prompt += "Steps:\n";
    fa_prompt += "1) source conda.sh; conda activate " + env_name + "\n";
    fa_prompt += "2) Confirm python is .../envs/" + env_name + "/bin/python and torch.cuda works\n";
    fa_prompt += "3) If import flash_attn works AND version matches AND force_reinstall=false: skip build\n";
    fa_prompt += "4) Else compile install:\n";
    fa_prompt += "     pip install flash-attn==" + flash_version + " --no-build-isolation\n";
    fa_prompt += "   If that fails, try: FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==" + flash_version + " --no-build-isolation\n";
    fa_prompt += "   Wait for ninja/nvcc build to finish (can take many minutes).\n";
    fa_prompt += "5) python -c \"import flash_attn; print(flash_attn.__version__)\"\n\n";
    fa_prompt += "ok=true only if import flash_attn succeeds.\n";
    fa_prompt += "Put version in version. Include last ~40 lines of build log in log_tail.\n";

    fa_r = agent(fa_prompt, #{
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
}

// =============================================================================
phase("Verify");
// =============================================================================

let v_prompt = "";
v_prompt += "FINAL verification: flash_attn + lm_eval must work for the MxMoE project.\n";
v_prompt += "Activate conda env `" + env_name + "`.\n";
v_prompt += "export CUDA_VISIBLE_DEVICES=" + gpu_devices + "\n";
v_prompt += "export CUDA_HOME=" + cuda_home + "\n";
v_prompt += "export PATH=$CUDA_HOME/bin:$PATH\n";
v_prompt += "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n";
v_prompt += "export PYTHONPATH=<repo_root>:$PYTHONPATH\n";
v_prompt += "Repo: use git rev-parse --show-toplevel or workspace MxMoE root.\n\n";
v_prompt += "### A) flash_attn CUDA smoke (skip if skip_flash was true and package missing — else required)\n";
v_prompt += "import flash_attn; from flash_attn import flash_attn_func\n";
v_prompt += "q,k,v = torch.randn(1,128,8,64,device='cuda',dtype=torch.float16) each\n";
v_prompt += "out = flash_attn_func(q,k,v,causal=True); assert out.shape==(1,128,8,64)\n";
v_prompt += "from transformers.utils import is_flash_attn_2_available; assert is_flash_attn_2_available()\n";
v_prompt += "Print FLASH_ATTN PASS\n\n";
v_prompt += "### B) lm_eval + project Evaluator (skip only if skip_lm_eval)\n";
v_prompt += "import lm_eval\n";
v_prompt += "from lm_eval import tasks\n";
v_prompt += "from lm_eval.models.huggingface import HFLM\n";
v_prompt += "from mxmoe.quant.evaluator import Evaluator\n";
v_prompt += "Try tasks.get_task_dict(['arc_easy','piqa','hellaswag','winogrande'])\n";
v_prompt += "  if API fails, use TaskManager and match_tasks — still ok if tasks resolve\n";
v_prompt += "Print LM_EVAL PASS and PROJECT PASS\n\n";
v_prompt += "### C) Project usage note\n";
v_prompt += "Grep confirms quant.py / moe_tracer / gptq use attn_implementation=flash_attention_2\n";
v_prompt += "and evaluator.eval_tasks uses lm_eval — report that in details.\n\n";
v_prompt += "Set flash_ok / lm_eval_ok / project_ok booleans.\n";
v_prompt += "ok=true only if required packages (non-skipped) all pass.\n";

let ver = agent(v_prompt, #{
    label: "verify-both",
    capability_mode: "all",
    output_schema: verify_schema,
});

if ver == () || !ver.success || ver.output == () {
    pause("verification", "Verify agent failed to return structured result.");
}
if ver.output.ok != true {
    pause("verification", "Verify FAILED: " + ver.output.summary + " | " + ver.output.details);
}
log("Verify PASS: " + ver.output.summary);

// =============================================================================
phase("Report");
// =============================================================================

let rep_prompt = "";
rep_prompt += "Write a short markdown report of the flash_attn + lm_eval install for MxMoE.\n";
rep_prompt += "env_name: " + env_name + "\n";
rep_prompt += "cuda_home: " + cuda_home + "\n";
rep_prompt += "gpu: " + gpu_devices + "\n";
rep_prompt += "precheck: " + pre.output.summary + "\n";
if lm_r != () && lm_r.success && lm_r.output != () {
    rep_prompt += "lm_eval install: " + lm_r.output.summary + " ver=" + lm_r.output.version + "\n";
}
if fa_r != () && fa_r.success && fa_r.output != () {
    rep_prompt += "flash_attn install: " + fa_r.output.summary + " ver=" + fa_r.output.version + "\n";
}
rep_prompt += "verify: " + ver.output.summary + "\n";
rep_prompt += "flash_ok=" + ver.output.flash_ok.to_string() + " lm_eval_ok=" + ver.output.lm_eval_ok.to_string() + "\n\n";
rep_prompt += "Include: how to activate env, flash version, lm_eval version, note that quant/eval use FA2 + lm_eval.\n";
rep_prompt += "Do NOT write under repo docs/ (project AGENTS.md: no install/env reports in docs/).\n";
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
scratch += "- env: " + env_name + "\n";
scratch += "- flash_ok: " + ver.output.flash_ok.to_string() + "\n";
scratch += "- lm_eval_ok: " + ver.output.lm_eval_ok.to_string() + "\n";
scratch += "- report: " + report_path + "\n";
let sp = write_scratch_file("install-flash-lmeval-result.md", scratch);

complete(#{
    summary: "flash_attn + lm_eval installed and verified for MxMoE",
    env_name: env_name,
    flash_ok: ver.output.flash_ok,
    lm_eval_ok: ver.output.lm_eval_ok,
    project_ok: ver.output.project_ok,
    flash_version: ver.output.flash_version,
    lm_eval_version: ver.output.lm_eval_version,
    report_path: report_path,
    scratch: sp,
});
