// reproduce-acc: Generic MxMoE paper accuracy pipeline (PPL + optional
// zero-shot). Model is a REQUIRED parameter (model_id + model_path) — not
// hard-coded to any single checkpoint. Scope excludes kernel / run_mxmoe_gg /
// Fig.5. Pipeline: ID2NAME → real gate → gptq-had calib → bits_solver →
// Baseline + MxMoE eval. Side effects: calib/, qconfigs/, data/cache/, GPU.
// Reports: workflow scratch only — never docs/ (see AGENTS.md).
//
// Required args:
//   model_id          project_config ID2NAME key (e.g. "qwen2_moe", "ds2", "mixtral")
//   model_path        absolute local weights directory for that model
//
// Optional args:
//   env_name          default "mxmoe"
//   gpu_devices       default "2,3"
//   cuda_home         default "/usr/local/cuda-12.4"
//   nsamples          default "128"
//   seqlen            default "4096"
//   dataset           default "wiki2"   (gate/calib dataset id used by tracer)
//   qtype             default "gptq-had"
//   filter_a          default "w4a4_g-1_sym"   (first solver/calib scheme)
//   filter_b          default "w8a8_g-1_sym"   (second scheme)
//   wbits             default "5.0"
//   r                 default "0.75"
//   solver_batch      default "8192"
//   run_zeroshot      default true
//   run_uniform_w4a4  default false
//   paper_note        default ""  (optional human note / paper refs for report)
//   skip_model_path   default false
//   skip_gate         default false
//   skip_calib        default false
//   skip_solver       default false
//   skip_baseline     default false
//   skip_mxmoe_eval   default false
//
let meta = #{
    name: "reproduce-acc",
    description: "Generic MoE accuracy reproduction: parameterized model_id/path, gate, gptq-had calib, ILP, Baseline+MxMoE PPL and optional zero-shot (no kernel)",
    when_to_use: "When you need Table-1 style PPL/zero-shot for any supported ID2NAME MoE model, with model_id and model_path passed as args",
    phases: [
        #{ title: "Precheck", detail: "env, GPU, flash_attn, lm_eval, model path exists" },
        #{ title: "Model-Path", detail: "point project_config.ID2NAME[model_id] at model_path" },
        #{ title: "Gate-Trace", detail: "real moe-gate on dataset" },
        #{ title: "Calib", detail: "gptq-had layer_out_norm for filter_a and filter_b" },
        #{ title: "Align-Loss", detail: "calib JSON paths match EXPERT_QUANT_LOSS" },
        #{ title: "Solver", detail: "bits_solver mixed qconfig + tile_cfg" },
        #{ title: "Eval-Baseline", detail: "baseline PPL (+ zero-shot)" },
        #{ title: "Eval-MxMoE", detail: "MxMoE mixed qconfig PPL (+ zero-shot)" },
        #{ title: "Eval-Optional", detail: "optional uniform filter_a control" },
        #{ title: "Report", detail: "scratch summary (no docs/)" },
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

// Required: model_id + model_path (no hard-coded model defaults)
if args == () {
    pause("infra", "Pass args.model_id and args.model_path (required). Example: model_id=qwen2_moe, model_path=/path/to/weights");
}
let model_id_raw = args.model_id;
let model_path_raw = args.model_path;
if model_id_raw == () || type_of(model_id_raw) != "string" || model_id_raw == "" {
    pause("infra", "args.model_id is required (e.g. qwen2_moe | ds2 | mixtral | qwen2_moe_57b).");
}
if model_path_raw == () || type_of(model_path_raw) != "string" || model_path_raw == "" {
    pause("infra", "args.model_path is required (absolute local checkpoint directory).");
}
let model_id = model_id_raw;
let model_path = model_path_raw;

let env_name = arg_str(args, "env_name", "mxmoe");
let gpu_devices = arg_str(args, "gpu_devices", "2,3");
let cuda_home = arg_str(args, "cuda_home", "/usr/local/cuda-12.4");
let nsamples = arg_str(args, "nsamples", "128");
let seqlen = arg_str(args, "seqlen", "4096");
let dataset = arg_str(args, "dataset", "wiki2");
let qtype = arg_str(args, "qtype", "gptq-had");
let filter_a = arg_str(args, "filter_a", "w4a4_g-1_sym");
let filter_b = arg_str(args, "filter_b", "w8a8_g-1_sym");
let wbits = arg_str(args, "wbits", "5.0");
let r_val = arg_str(args, "r", "0.75");
let solver_batch = arg_str(args, "solver_batch", "8192");
let run_zeroshot = arg_bool(args, "run_zeroshot", true);
let run_uniform_w4a4 = arg_bool(args, "run_uniform_w4a4", false);
let paper_note = arg_str(args, "paper_note", "");
let skip_model_path = arg_bool(args, "skip_model_path", false);
let skip_gate = arg_bool(args, "skip_gate", false);
let skip_calib = arg_bool(args, "skip_calib", false);
let skip_solver = arg_bool(args, "skip_solver", false);
let skip_baseline = arg_bool(args, "skip_baseline", false);
let skip_mxmoe_eval = arg_bool(args, "skip_mxmoe_eval", false);

let step_schema = #{
    "type": "object",
    "required": ["ok", "summary", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "details": #{ "type": "string" },
        "artifact_path": #{ "type": "string" },
        "log_tail": #{ "type": "string" },
    },
};

let eval_schema = #{
    "type": "object",
    "required": ["ok", "summary", "ppl", "details"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "ppl": #{ "type": "number" },
        "acc_avg": #{ "type": "number" },
        "zero_shot_json": #{ "type": "string" },
        "details": #{ "type": "string" },
        "artifact_path": #{ "type": "string" },
    },
};

let preamble = "";
preamble += "Always use isolated conda env `" + env_name + "` (not base/kitty).\n";
preamble += "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate " + env_name + "\n";
preamble += "export CUDA_VISIBLE_DEVICES=" + gpu_devices + "\n";
preamble += "export CUDA_HOME=" + cuda_home + "\n";
preamble += "export PATH=$CUDA_HOME/bin:$PATH\n";
preamble += "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n";
preamble += "cd to MxMoE repo root (git rev-parse --show-toplevel).\n";
preamble += "export PYTHONPATH=$PWD:$PYTHONPATH\n";
preamble += "Confirm: which python is .../envs/" + env_name + "/bin/python\n";
preamble += "Target model_id=" + model_id + " model_path=" + model_path + "\n";

// =============================================================================
phase("Precheck");
// =============================================================================

let pre_p = "";
pre_p += "Precheck for MxMoE accuracy reproduction (PPL + optional zero-shot; NO kernel bench).\n";
pre_p += preamble;
pre_p += "\nRun shell checks (do not guess):\n";
pre_p += "1) conda env " + env_name + " exists; torch.cuda.is_available()\n";
pre_p += "2) import flash_attn; transformers.utils.is_flash_attn_2_available()\n";
pre_p += "3) import lm_eval; from mxmoe.quant.evaluator import Evaluator\n";
pre_p += "4) test -d \"" + model_path + "\" and list config/weights under it\n";
pre_p += "5) project_config.ID2NAME must eventually support key \"" + model_id + "\" (exists in dict after Model-Path step)\n";
pre_p += "6) gurobipy import works (needed for bits_solver)\n";
pre_p += "7) df -h . (warn if free < 50GB)\n\n";
pre_p += "ok=true only if env+torch+cuda+model_dir+flash_attn+lm_eval OK.\n";

let pre = agent(pre_p, #{
    label: "precheck",
    capability_mode: "all",
    output_schema: step_schema,
});
if pre == () || !pre.success || pre.output == () {
    pause("infra", "Precheck agent failed.");
}
if pre.output.ok != true {
    pause("infra", "Precheck not-ok: " + pre.output.summary + " | " + pre.output.details);
}
log("Precheck OK: " + pre.output.summary);

// =============================================================================
phase("Model-Path");
// =============================================================================

if skip_model_path {
    log("skip_model_path=true — ID2NAME[\"" + model_id + "\"] must already point at " + model_path);
} else {
    let mp = "";
    mp += "Update project_config.py ID2NAME for the parameterized model.\n";
    mp += preamble;
    mp += "Set ID2NAME[\"" + model_id + "\"] = \"" + model_path + "\"\n";
    mp += "Steps:\n";
    mp += "1) Read project_config.py\n";
    mp += "2) If key missing, add it; if present, replace path only for this key\n";
    mp += "3) Do not change other model entries unless required for syntax\n";
    mp += "4) python -c \"from project_config import ID2NAME; print(ID2NAME['" + model_id + "'])\"\n";
    mp += "5) Prefer: AutoTokenizer.from_pretrained(that path, trust_remote_code=True)\n\n";
    mp += "ok=true only if printed path matches model_path and load succeeds (or path exists if load too heavy — prefer tokenizer load).\n";

    let mpr = agent(mp, #{
        label: "model-path",
        capability_mode: "all",
        output_schema: step_schema,
    });
    if mpr == () || !mpr.success || mpr.output == () || mpr.output.ok != true {
        let d = "model-path failed";
        if mpr != () && mpr.success && mpr.output != () {
            d = mpr.output.summary + " | " + mpr.output.details;
        }
        pause("infra", d);
    }
    log("Model path OK: " + mpr.output.summary);
}

// =============================================================================
phase("Gate-Trace");
// =============================================================================

if skip_gate {
    log("skip_gate=true — expect calib/gate/" + model_id + "/" + dataset + "/" + seqlen + "/moe-gate.json");
} else {
    let gp = "";
    gp += "Collect REAL MoE gate trace for model_id=" + model_id + ".\n";
    gp += preamble;
    gp += "Command:\n";
    gp += "python -m mxmoe.quant.moe_tracer \\\n";
    gp += "  --model " + model_id + " --trace_gate --dataset " + dataset + " \\\n";
    gp += "  --seqlen " + seqlen + " --nsamples " + nsamples + "\n\n";
    gp += "Wait for full model load + exit 0.\n";
    gp += "Pass: calib/gate/" + model_id + "/" + dataset + "/" + seqlen + "/moe-gate.json exists with access_freq.\n";
    gp += "ok=true only if pass. artifact_path = json path.\n";

    let gr = agent(gp, #{
        label: "gate-trace",
        capability_mode: "all",
        output_schema: step_schema,
    });
    if gr == () || !gr.success || gr.output == () || gr.output.ok != true {
        let d = "gate failed";
        if gr != () && gr.success && gr.output != () {
            d = gr.output.summary + " | " + gr.output.details;
        }
        pause("infra", d);
    }
    log("Gate OK: " + gr.output.summary);
}

// =============================================================================
phase("Calib");
// =============================================================================

if skip_calib {
    log("skip_calib=true");
} else {
    let c1p = "";
    c1p += "Quant-loss calibration for scheme filter_a=" + filter_a + " model=" + model_id + ".\n";
    c1p += preamble;
    c1p += "python -m mxmoe.quant.quant calib \\\n";
    c1p += "  --model " + model_id + " --method " + qtype + " \\\n";
    c1p += "  --metric layer_out_norm --qcfg " + filter_a + " \\\n";
    c1p += "  --nsamples " + nsamples + "\n\n";
    c1p += "Long GPU job. ok=true if *layer_out_norm*.json for this scheme appears under calib/.\n";
    c1p += "artifact_path = exact output path.\n";

    let c1 = agent(c1p, #{
        label: "calib-a",
        capability_mode: "all",
        output_schema: step_schema,
    });
    if c1 == () || !c1.success || c1.output == () || c1.output.ok != true {
        let d = "calib filter_a failed";
        if c1 != () && c1.success && c1.output != () {
            d = c1.output.summary + " | " + c1.output.details;
        }
        pause("infra", d);
    }
    log("Calib A OK: " + c1.output.summary);

    let c2p = "";
    c2p += "Quant-loss calibration for scheme filter_b=" + filter_b + " model=" + model_id + ".\n";
    c2p += preamble;
    c2p += "python -m mxmoe.quant.quant calib \\\n";
    c2p += "  --model " + model_id + " --method " + qtype + " \\\n";
    c2p += "  --metric layer_out_norm --qcfg " + filter_b + " \\\n";
    c2p += "  --nsamples " + nsamples + "\n\n";
    c2p += "ok=true if layer_out_norm json for filter_b under calib/. artifact_path = path.\n";

    let c2 = agent(c2p, #{
        label: "calib-b",
        capability_mode: "all",
        output_schema: step_schema,
    });
    if c2 == () || !c2.success || c2.output == () || c2.output.ok != true {
        let d = "calib filter_b failed";
        if c2 != () && c2.success && c2.output != () {
            d = c2.output.summary + " | " + c2.output.details;
        }
        pause("infra", d);
    }
    log("Calib B OK: " + c2.output.summary);
}

// =============================================================================
phase("Align-Loss");
// =============================================================================

let alp = "";
alp += "Align calib JSON paths with EXPERT_QUANT_LOSS for this model_id.\n";
alp += preamble;
alp += "model_id=" + model_id + " qtype=" + qtype + " schemes: " + filter_a + ", " + filter_b + "\n";
alp += "1) Read EXPERT_QUANT_LOSS[qtype][model_id] expected paths (if key missing, create entries or fix config)\n";
alp += "2) Find actual calib outputs under calib/\n";
alp += "3) symlink/copy or patch EXPERT_QUANT_LOSS so both schemes open successfully\n";
alp += "4) Verify both files open in Python\n";
alp += "ok=true only if both schemes resolve.\n";

let al = agent(alp, #{
    label: "align-loss",
    capability_mode: "all",
    output_schema: step_schema,
});
if al == () || !al.success || al.output == () || al.output.ok != true {
    let d = "align-loss failed";
    if al != () && al.success && al.output != () {
        d = al.output.summary + " | " + al.output.details;
    }
    pause("infra", d);
}
log("Align loss OK: " + al.output.summary);

// =============================================================================
phase("Solver");
// =============================================================================

let qconfig_hint = "qconfigs/" + filter_a + "+" + filter_b + "/" + model_id + "_" + qtype + "_Slayer_bs" + solver_batch + "_wbits" + wbits + "_r" + r_val + ".json";

if skip_solver {
    log("skip_solver=true — expect qconfig for model " + model_id + " near " + qconfig_hint);
} else {
    let sp = "";
    sp += "ILP bits_solver for model_id=" + model_id + " (mixed precision qconfig).\n";
    sp += preamble;
    sp += "Ensure calib/perf/performance_table.json exists (copy from perf/ if needed).\n";
    sp += "Command:\n";
    sp += "python -m mxmoe.quant.bits_solver \\\n";
    sp += "  --model " + model_id + " --qtype " + qtype + " \\\n";
    sp += "  --wbits " + wbits + " --solve_mode layer --batch " + solver_batch + " \\\n";
    sp += "  --r " + r_val + " \\\n";
    sp += "  --filter_list " + filter_a + " " + filter_b + " \\\n";
    sp += "  --trace_file calib/gate/" + model_id + "/" + dataset + "/" + seqlen + "/moe-gate.json \\\n";
    sp += "  --perf_file calib/perf/performance_table.json\n\n";
    sp += "ok=true if qconfig json exists (not only tile_cfg). artifact_path = qconfig path.\n";

    let sr = agent(sp, #{
        label: "bits-solver",
        capability_mode: "all",
        output_schema: step_schema,
    });
    if sr == () || !sr.success || sr.output == () || sr.output.ok != true {
        let d = "solver failed";
        if sr != () && sr.success && sr.output != () {
            d = sr.output.summary + " | " + sr.output.details;
        }
        pause("infra", d);
    }
    log("Solver OK: " + sr.output.summary);
    if sr.output.artifact_path != () && sr.output.artifact_path != "" {
        qconfig_hint = sr.output.artifact_path;
    }
}

let zs_tasks = "arc_challenge arc_easy hellaswag lambada_openai lambada_standard piqa winogrande";
let task_args = "ppl";
if run_zeroshot {
    task_args = "ppl " + zs_tasks;
}

// =============================================================================
phase("Eval-Baseline");
// =============================================================================

let base_eval = ();
if skip_baseline {
    log("skip_baseline=true");
} else {
    let ep = "";
    ep += "Evaluate BASELINE for model_id=" + model_id + " (PPL";
    if run_zeroshot {
        ep += " + zero-shot";
    }
    ep += ").\n";
    ep += preamble;
    if paper_note != "" {
        ep += "Optional paper_note from args: " + paper_note + "\n";
    }
    ep += "tasks: " + task_args + "\n";
    ep += "Prefer full-precision / no-quant path via quant.py eval or Evaluator.eval_ppl after load_hf_model.\n";
    ep += "Attempt sensible baselines for THIS model; do not assume a single HF name.\n";
    ep += "ok=true if finite PPL > 1. Record ppl, optional acc_avg.\n";

    base_eval = agent(ep, #{
        label: "eval-baseline",
        capability_mode: "all",
        output_schema: eval_schema,
    });
    if base_eval == () || !base_eval.success || base_eval.output == () || base_eval.output.ok != true {
        let d = "baseline eval failed";
        if base_eval != () && base_eval.success && base_eval.output != () {
            d = base_eval.output.summary + " | " + base_eval.output.details;
        }
        pause("verification", d);
    }
    log("Baseline eval OK: " + base_eval.output.summary);
}

// =============================================================================
phase("Eval-MxMoE");
// =============================================================================

let mx_eval = ();
if skip_mxmoe_eval {
    log("skip_mxmoe_eval=true");
} else {
    let ep2 = "";
    ep2 += "Evaluate MxMoE mixed qconfig for model_id=" + model_id + ".\n";
    ep2 += preamble;
    ep2 += "qconfig hint: " + qconfig_hint + "\n";
    ep2 += "Resolve real path: find qconfigs -name '*" + model_id + "*wbits" + wbits + "*r" + r_val + "*.json' ! -name '*tile*'\n";
    ep2 += "python -m mxmoe.quant.quant eval --model " + model_id + " --method " + qtype + " \\\n";
    ep2 += "  --qconfig <resolved.json> --tasks " + task_args + "\n\n";
    if paper_note != "" {
        ep2 += "Optional paper_note: " + paper_note + "\n";
    }
    ep2 += "ok=true if finite PPL. artifact_path = qconfig used.\n";

    mx_eval = agent(ep2, #{
        label: "eval-mxmoe",
        capability_mode: "all",
        output_schema: eval_schema,
    });
    if mx_eval == () || !mx_eval.success || mx_eval.output == () || mx_eval.output.ok != true {
        let d = "mxmoe eval failed";
        if mx_eval != () && mx_eval.success && mx_eval.output != () {
            d = mx_eval.output.summary + " | " + mx_eval.output.details;
        }
        pause("verification", d);
    }
    log("MxMoE eval OK: " + mx_eval.output.summary);
}

// =============================================================================
phase("Eval-Optional");
// =============================================================================

let uni_eval = ();
if !run_uniform_w4a4 {
    log("run_uniform_w4a4=false");
} else {
    let up = "";
    up += "Optional uniform control using filter_a=" + filter_a + " for model " + model_id + ".\n";
    up += preamble;
    up += "python -m mxmoe.quant.quant eval --model " + model_id + " --method " + qtype + " \\\n";
    up += "  --qstr " + filter_a + " --tasks " + task_args + "\n";
    up += "ok=true if finite PPL.\n";

    uni_eval = agent(up, #{
        label: "eval-uniform",
        capability_mode: "all",
        output_schema: eval_schema,
    });
    if uni_eval == () || !uni_eval.success || uni_eval.output == () || uni_eval.output.ok != true {
        log("uniform eval failed (optional)");
    } else {
        log("Uniform OK: " + uni_eval.output.summary);
    }
}

// =============================================================================
phase("Report");
// =============================================================================

let rep = "";
rep += "Write markdown accuracy report for parameterized MoE model (PPL/ZS only).\n";
rep += "Do NOT write under repo docs/. Put full markdown in details; ok=true.\n";
rep += "model_id=" + model_id + " model_path=" + model_path + "\n";
rep += "filter_a=" + filter_a + " filter_b=" + filter_b + " wbits=" + wbits + " r=" + r_val + "\n";
rep += "qconfig_hint=" + qconfig_hint + "\n";
if paper_note != "" {
    rep += "paper_note=" + paper_note + "\n";
}
if base_eval != () && base_eval.success && base_eval.output != () {
    rep += "baseline: " + base_eval.output.summary + " | " + base_eval.output.details + "\n";
}
if mx_eval != () && mx_eval.success && mx_eval.output != () {
    rep += "mxmoe: " + mx_eval.output.summary + " | " + mx_eval.output.details + "\n";
}
if uni_eval != () && uni_eval.success && uni_eval.output != () {
    rep += "uniform: " + uni_eval.output.summary + "\n";
}
rep += "align: " + al.output.summary + "\n";
rep += "Sections: config, artifacts, results table, skips, non-goals (no kernel).\n";

let repor = agent(rep, #{
    label: "report",
    capability_mode: "all",
    output_schema: step_schema,
});

let report_body = "# reproduce-acc\n\nmodel_id=" + model_id + "\n";
if repor != () && repor.success && repor.output != () && repor.output.details != () {
    report_body = repor.output.details;
}
let report_path = write_scratch_file("reproduce-acc-report.md", report_body);

let base_ppl = -1.0;
let mx_ppl = -1.0;
let base_avg = -1.0;
let mx_avg = -1.0;
if base_eval != () && base_eval.success && base_eval.output != () {
    base_ppl = base_eval.output.ppl;
    if base_eval.output.acc_avg != () {
        base_avg = base_eval.output.acc_avg;
    }
}
if mx_eval != () && mx_eval.success && mx_eval.output != () {
    mx_ppl = mx_eval.output.ppl;
    if mx_eval.output.acc_avg != () {
        mx_avg = mx_eval.output.acc_avg;
    }
}

complete(#{
    summary: "MoE accuracy pipeline finished for model_id=" + model_id + " (PPL/zero-shot; no kernel)",
    model_id: model_id,
    model_path: model_path,
    qconfig: qconfig_hint,
    baseline_ppl: base_ppl,
    mxmoe_ppl: mx_ppl,
    baseline_acc_avg: base_avg,
    mxmoe_acc_avg: mx_avg,
    run_zeroshot: run_zeroshot,
    report_scratch: report_path,
});
