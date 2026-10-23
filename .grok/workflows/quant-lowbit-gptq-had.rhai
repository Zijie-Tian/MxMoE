// quant-lowbit-gptq-had: Experiment-only online GPTQ-Had for the 12 solved
// mixed-precision qconfigs (ds2 and qwen2_moe; BPV 2.25 and 3.25; R=0, 0.5, 1).
// Does not re-solve ILP, re-run calibration, call export_gptq_hf, run PPL or
// zero-shot, or bench kernels. Shared environment is conda env mxmoe on CUDA 13.0
// (doc/environment.md). Does not install packages or use another env.
// One physical GPU per model; that model's six configs run one after another.
// Side effects: two GPUs, fake-quant HF trees under save_root. Reports are scratch only.
//
// Args (all optional; recipe defaults match the 2026-10-16 config package):
//   env_name        default "mxmoe" (any other value is rejected)
//   cuda_home       default "" (resolve CUDA 13.0; never 12.4 or newest cuda*)
//   config_root     default "outputs/lowbit_allocations_20261016/mxmoe_lowbit_configs_12"
//   save_root       default "/mnt/raid/tzj/mxmoe/lowbit_gptq_had_20261016"
//   gpu_ds2         default "0"     (only ds2; never share with qwen2_moe)
//   gpu_qwen        default "1"     (only qwen2_moe)
//   nsamples        default "128"   (do not use the historical n256 maps)
//   seed            default "42"
//   budgets         default "2.25,3.25"
//   r_values        default "0,0.5,1"
//   skip_existing   default true    (reuse a finished save dir; attach to a live same-qconfig process)
//   skip_quant      default false

let meta = #{
    name: "quant-lowbit-gptq-had",
    description: "Quant-only online GPTQ-Had in conda env mxmoe on CUDA 13.0 for the 12 solved ds2 and qwen2_moe low-bit qconfigs, one GPU per model, no ILP and no accuracy eval",
    when_to_use: "When the low-bit allocation package is already solved and those qconfigs should be quantized on the shared mxmoe stack",
    phases: [
        #{ title: "Precheck", detail: "mxmoe, CUDA 13.0, checkpoints, config checksums, two distinct GPUs" },
        #{ title: "Quant", detail: "ds2 on gpu_ds2 || qwen2_moe on gpu_qwen; six configs sequential per GPU" },
        #{ title: "Report", detail: "scratch summary of BPV, peaks, and save paths" },
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

fn quant_prompt(model_id, gpu_id, skip_note, preamble, budgets, r_values, nsamples, seed) {
    let p = "";
    p += "Quantize one model. One process. One physical GPU.\n";
    p += preamble;
    p += "model_id=" + model_id + "\n";
    p += "CUDA_VISIBLE_DEVICES=" + gpu_id + " exactly. Never list another GPU.\n";
    p += "budgets=" + budgets + " r_values=" + r_values + "\n";
    p += skip_note;
    p += "For each budget B in budgets and each R in r_values, in that order:\n";
    p += "  qconfig = <config_root>/" + model_id + "/bpv<B>/r<R>/qconfig.json\n";
    p += "  R directory names are r0, r0.5, r1 (not r0.0).\n";
    p += "  save_dir = <save_root>/" + model_id + "/bpv<B>_r<R>\n";
    p += "  log = <repo>/log/quant_lowbit_" + model_id + "_bpv<B>_r<R>_gpu" + gpu_id + ".log\n";
    p += "Command, from the repo root, with the mxmoe interpreter:\n";
    p += "  CUDA_VISIBLE_DEVICES=" + gpu_id + " conda run --no-capture-output -n mxmoe python -m mxmoe.quant.quant eval \\\n";
    p += "    --model " + model_id + " --method gptq-had \\\n";
    p += "    --qconfig <qconfig> --gptq-nsamples " + nsamples + " --seed " + seed + " \\\n";
    p += "    --device-map cpu --skip-eval \\\n";
    p += "    --save <save_dir>/quant_report.json --save-model <save_dir>\n";
    p += "Redirect stdout and stderr to the log above.\n";
    p += "After each config, read the log. Require the line 'Expert BPV' and '[quant] DONE skip-eval'.\n";
    p += "Compare logged effective BPV to summary.csv effective_bpv for that model, budget, and R. Mismatch means ok=false.\n";
    p += "Record peak_mem_gib from the log. CUDA OOM or a traceback means stop the remaining configs for this model and ok=false.\n";
    p += "Do not run PPL, zero-shot, or kernel timing. Fake-quant weights are not a speed result.\n";
    p += "summary: how many configs finished, skipped, or attached, and any failure.\n";
    p += "details: one line per config: budget, R, status, effective_bpv, peak_mem_gib, save_dir.\n";
    p += "artifact_path: save_root for this model.\n";
    p
}

let env_name = arg_str(args, "env_name", "mxmoe");
let cuda_home = arg_str(args, "cuda_home", "");
let config_root = arg_str(args, "config_root", "outputs/lowbit_allocations_20261016/mxmoe_lowbit_configs_12");
let save_root = arg_str(args, "save_root", "/mnt/raid/tzj/mxmoe/lowbit_gptq_had_20261016");
let gpu_ds2 = arg_str(args, "gpu_ds2", "0");
let gpu_qwen = arg_str(args, "gpu_qwen", "1");
let nsamples = arg_str(args, "nsamples", "128");
let seed = arg_str(args, "seed", "42");
let budgets = arg_str(args, "budgets", "2.25,3.25");
let r_values = arg_str(args, "r_values", "0,0.5,1");
let skip_existing = arg_bool(args, "skip_existing", true);
let skip_quant = arg_bool(args, "skip_quant", false);

if env_name != "mxmoe" {
    pause("infra", "Refusing env_name='" + env_name + "'. This quant workload uses conda env mxmoe only.");
}
if gpu_ds2 == gpu_qwen {
    pause("infra", "gpu_ds2 and gpu_qwen must be two different physical GPU ids. One GPU runs one model.");
}

let step_schema = #{
    "type": "object",
    "required": ["ok", "summary", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "details": #{ "type": "string" },
        "artifact_path": #{ "type": "string" },
        "cuda_release": #{ "type": "string" },
    },
};

let preamble = "";
preamble += "REQUIRED environment: conda env mxmoe only. See doc/environment.md. Do not install packages or use another env.\n";
preamble += "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate mxmoe\n";
preamble += "Python is that interpreter. Quant commands use conda run --no-capture-output -n mxmoe python -m ...\n";
preamble += "Resolve CUDA 13.0 before any GPU command. Supplied cuda_home arg: '" + cuda_home + "'.\n";
preamble += "If that arg is non-empty, it is the only candidate and its nvcc must be 13.0. Refuse 12.4 and 13.1.\n";
preamble += "If it is empty, do not export an empty CUDA_HOME and do not keep an inherited 12.4 or 13.1 value.\n";
preamble += "Use process CUDA_HOME only when its nvcc is 13.0; else /usr/local/cuda-13.0 when that nvcc is 13.0. Never pick the newest cuda directory.\n";
preamble += "Then export PATH=$CUDA_HOME/bin:$PATH and LD_LIBRARY_PATH from $CUDA_HOME/lib64 (or lib if lib64 is absent).\n";
preamble += "cd the MxMoE git toplevel (git rev-parse --show-toplevel). export PYTHONPATH=$PWD:$PYTHONPATH\n";
preamble += "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8\n";
preamble += "config_root=" + config_root + " (if relative, resolve under the repo root).\n";
preamble += "save_root=" + save_root + "\n";
preamble += "Recipe is fixed: method gptq-had, offline Hadamard, online Hadamard OFF, WikiText2 ";
preamble += nsamples + "x4096, seed " + seed + ", act_order true, static_groups true, percdamp 0.01, A16, asymmetric.\n";
preamble += "Do not pass --online_had. Do not pass --qweight. Do not load n256 prequant maps.\n";
preamble += "Do not edit qconfig bitwidths. Do not run bits_solver or calib. Do not call export_gptq_hf.\n";
preamble += "Read qconfig.json only. expert_weight_map.json is a check table, not the quant input.\n";
preamble += "DS2 dense layer 0 must stay W4A16 group -1 asymmetric; the quant entry asserts that.\n";
preamble += "A budget is an upper bound. If summary.csv effective_bpv is below the budget, that is a legal solve.\n";
preamble += "Place weights on CPU and move one decoder layer to the visible GPU (--device-map cpu). Do not use accelerate fit-map with --qconfig.\n";

// =============================================================================
phase("Precheck");
// =============================================================================

let pre_p = "";
pre_p += "Precheck only. Do not start quantization.\n";
pre_p += preamble;
pre_p += "Run these checks with the shell. ok=true only if every one exits 0.\n";
pre_p += "1) mxmoe only: Python 3.12, torch 2.13.0+cu130, torch.version.cuda 13.0, nvcc release 13.0, transformers 5.19.0, fast_hadamard_transform imports. Do not pip install. Set cuda_release to that nvcc release.\n";
pre_p += "2) conda run --no-capture-output -n mxmoe python project_config.py --check ds2 qwen2_moe exits 0. Do not rewrite .env.\n";
pre_p += "3) config_root contains README.txt, provenance.json, summary.csv, SHA256SUMS, and these 12 files:\n";
pre_p += "   ds2 and qwen2_moe; bpv2.25 and bpv3.25; r0, r0.5, r1; each with qconfig.json and expert_weight_map.json.\n";
pre_p += "4) In config_root run: sha256sum -c SHA256SUMS. All listed files must be OK.\n";
pre_p += "5) nvidia-smi: GPU " + gpu_ds2 + " and GPU " + gpu_qwen + " exist and are different devices.\n";
pre_p += "   Record used/free MiB. A busy GPU is allowed when the only compute process is an existing mxmoe.quant.quant for that same model.\n";
pre_p += "6) Confirm mxmoe/quant/quant.py eval accepts --gptq-nsamples, --device-map, --skip-eval, and --save-model.\n";
pre_p += "summary: one line. details: checksum result, both checkpoint paths, both GPU free MiB, resolved save_root.\n";
pre_p += "artifact_path: the resolved save_root. cuda_release must be 13.0 or ok=false.\n";

let pre = agent(pre_p, #{
    label: "precheck",
    capability_mode: "all",
    output_schema: step_schema,
});
if pre == () || !pre.success || pre.output == () || pre.output.ok != true {
    let why = "Precheck failed.";
    if pre != () && pre.success && pre.output != () && pre.output.summary != () {
        why = pre.output.summary;
    }
    pause("infra", why);
}
if pre.output.cuda_release != "13.0" {
    pause("infra", "Precheck CUDA is not 13.0. Refusing 12.4, 13.1, or an unresolved toolkit.");
}
log("Precheck OK: " + pre.output.summary);

// =============================================================================
phase("Quant");
// =============================================================================

let ds2_summary = "skipped";
let qwen_summary = "skipped";
let ds2_details = "";
let qwen_details = "";
let quant_ok = true;

if skip_quant {
    log("skip_quant=true");
} else {
    let skip_note = "skip_existing=false: rerun every config even if a save dir exists.\n";
    if skip_existing {
        skip_note = "skip_existing=true.\n";
        skip_note += "Finished: if <save_dir>/quant_report.json exists AND at least one model*.safetensors exists, do not rerun. Record status=skipped.\n";
        skip_note += "In progress: if a live process command line contains both this model id and this qconfig path, do not start a second process on that GPU. Tail its stdout log until it exits, then judge the save dir. Record status=attached.\n";
    }

    let jobs = [];
    jobs.push(#{
        prompt: quant_prompt("ds2", gpu_ds2, skip_note, preamble, budgets, r_values, nsamples, seed),
        label: "quant-ds2",
        capability_mode: "all",
        output_schema: step_schema,
    });
    jobs.push(#{
        prompt: quant_prompt("qwen2_moe", gpu_qwen, skip_note, preamble, budgets, r_values, nsamples, seed),
        label: "quant-qwen2-moe",
        capability_mode: "all",
        output_schema: step_schema,
    });
    let results = parallel(jobs);

    let labels = ["ds2", "qwen2_moe"];
    let i = 0;
    for slot in results {
        let name = labels[i];
        if slot == () || !slot.success || slot.output == () || slot.output.ok != true {
            quant_ok = false;
            let msg = name + " quant failed";
            if slot != () && slot.success && slot.output != () && slot.output.summary != () {
                msg = slot.output.summary;
            }
            if name == "ds2" { ds2_summary = msg; }
            if name == "qwen2_moe" { qwen_summary = msg; }
        } else {
            if name == "ds2" {
                ds2_summary = slot.output.summary;
                if slot.output.details != () { ds2_details = slot.output.details; }
            }
            if name == "qwen2_moe" {
                qwen_summary = slot.output.summary;
                if slot.output.details != () { qwen_details = slot.output.details; }
            }
        }
        i += 1;
    }
    if !quant_ok {
        pause("infra", "Quant failed. ds2: " + ds2_summary + " | qwen2_moe: " + qwen_summary);
    }
    log("Quant OK. ds2: " + ds2_summary + " | qwen2_moe: " + qwen_summary);
}

// =============================================================================
phase("Report");
// =============================================================================

let report = "";
report += "# quant-lowbit-gptq-had\n\n";
report += "Scope: online GPTQ-Had fake-quant only. No ILP, no calibration, no PPL, no kernel bench.\n\n";
report += "config_root: " + config_root + "\n\n";
report += "env: mxmoe  CUDA: 13.0\n\n";
if pre.output.artifact_path != () && pre.output.artifact_path != "" {
    report += "save_root: " + pre.output.artifact_path + "\n\n";
} else if save_root != "" {
    report += "save_root: " + save_root + "\n\n";
} else {
    report += "save_root: $MXMOE_LOWBIT_SAVE_ROOT\n\n";
}
report += "gpus: ds2=" + gpu_ds2 + " qwen2_moe=" + gpu_qwen + "\n\n";
report += "nsamples: " + nsamples + " seed: " + seed + "\n\n";
report += "budgets: " + budgets + " R: " + r_values + "\n\n";
report += "## Precheck\n\n" + pre.output.summary + "\n\n";
if pre.output.details != () {
    report += pre.output.details + "\n\n";
}
report += "## ds2\n\n" + ds2_summary + "\n\n" + ds2_details + "\n\n";
report += "## qwen2_moe\n\n" + qwen_summary + "\n\n" + qwen_details + "\n\n";
report += "Runtime cost tables in the allocation package are not measured speed.\n";

let report_path = write_scratch_file("quant-lowbit-gptq-had.md", report);
complete(#{
    summary: "ds2: " + ds2_summary + " | qwen2_moe: " + qwen_summary,
    report_path: report_path,
    save_root: save_root,
    quant_ok: quant_ok,
});
