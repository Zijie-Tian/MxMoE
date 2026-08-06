// reproduce-acc: Generic MxMoE paper accuracy pipeline (PPL + optional
// zero-shot). Model is REQUIRED (model_id + model_path). NO kernel/run_mxmoe_gg.
//
// MULTI-GPU: pin each heavy job to ONE physical GPU (do not share
// CUDA_VISIBLE_DEVICES across concurrent full-model loads). Default layout for
// three cards gpu_a/b/c (e.g. 1/2/3):
//   - Gate: gpu_a
//   - Parallel after gate: calib filter_a @ gpu_a | calib filter_b @ gpu_b |
//     baseline eval @ gpu_c (if run and gpu_c set)
//   - Align + Solver: CPU (any)
//   - Parallel after solver: MxMoE eval @ gpu_a | optional uniform @ gpu_b
//
// Side effects: calib/, qconfigs/, data/cache/, GPU. Reports: scratch only.
//
// Required args:
//   model_id, model_path
//
// Optional args:
//   env_name          default "mxmoe"
//   gpu_a             default "1"   (single-GPU pin for job A)
//   gpu_b             default "2"
//   gpu_c             default "3"   (empty string disables 3-way baseline||calib)
//   gpu_devices       default "1,2,3"  (documentation / precheck visibility)
//   cuda_home         default "/usr/local/cuda-12.4"
//   nsamples          default "128"
//   seqlen            default "4096"
//   dataset           default "wiki2"
//   qtype             default "gptq-had"
//   filter_a          default "w4a4_g-1_sym"
//   filter_b          default "w8a8_g-1_sym"
//   wbits             default "5.0"
//   r                 default "0.75"
//   solver_batch      default "8192"
//   run_zeroshot      default true
//   run_uniform_w4a4  default false
//   paper_note        default ""
//   skip_model_path / skip_gate / skip_calib / skip_solver /
//   skip_baseline / skip_mxmoe_eval  default false
//
let meta = #{
    name: "reproduce-acc",
    description: "Multi-GPU MoE accuracy reproduction: model_id/path args; parallel calib on separate GPUs; Baseline+MxMoE PPL/ZS (no kernel)",
    when_to_use: "When you need Table-1 style PPL/zero-shot for any ID2NAME MoE model and should utilize multiple GPUs in parallel",
    phases: [
        #{ title: "Precheck", detail: "env, multi-GPU visible, flash_attn, lm_eval, model path" },
        #{ title: "Model-Path", detail: ".env MXMOE_MODEL_<ID> = model_path" },
        #{ title: "Gate-Trace", detail: "real moe-gate on gpu_a" },
        #{ title: "Parallel-Calib", detail: "filter_a@gpu_a || filter_b@gpu_b (|| baseline@gpu_c)" },
        #{ title: "Align-Loss", detail: "calib paths match EXPERT_QUANT_LOSS" },
        #{ title: "Solver", detail: "bits_solver ILP (CPU)" },
        #{ title: "Parallel-Eval", detail: "MxMoE@gpu_a || optional uniform@gpu_b; baseline if not earlier" },
        #{ title: "Report", detail: "scratch summary only" },
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

if args == () {
    pause("infra", "Pass args.model_id and args.model_path (required).");
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
let gpu_a = arg_str(args, "gpu_a", "1");
let gpu_b = arg_str(args, "gpu_b", "2");
let gpu_c = arg_str(args, "gpu_c", "3");
let gpu_devices = arg_str(args, "gpu_devices", "1,2,3");
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

// Whether to fan baseline into the calib parallel panel (needs a distinct gpu_c)
let baseline_with_calib = !skip_baseline && gpu_c != "" && gpu_c != gpu_a && gpu_c != gpu_b;

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

// Base env (no multi-GPU export — each job pins one id)
let preamble = "";
preamble += "Use isolated conda env `" + env_name + "` only.\n";
preamble += "source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate " + env_name + "\n";
preamble += "export CUDA_HOME=" + cuda_home + "\n";
preamble += "export PATH=$CUDA_HOME/bin:$PATH\n";
preamble += "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\n";
preamble += "cd MxMoE repo root (git rev-parse --show-toplevel); export PYTHONPATH=$PWD:$PYTHONPATH\n";
preamble += "CRITICAL multi-GPU rule: set CUDA_VISIBLE_DEVICES to EXACTLY ONE physical GPU id for this job.\n";
preamble += "Do NOT export CUDA_VISIBLE_DEVICES=" + gpu_devices + " for concurrent full-model jobs.\n";
preamble += "model_id=" + model_id + " model_path=" + model_path + "\n";

let zs_tasks = "arc_challenge arc_easy hellaswag lambada_openai lambada_standard piqa winogrande";
let task_args = "ppl";
if run_zeroshot {
    task_args = "ppl " + zs_tasks;
}

// =============================================================================
phase("Precheck");
// =============================================================================

let pre_p = "";
pre_p += "Precheck multi-GPU accuracy pipeline (PPL/ZS; no kernel).\n";
pre_p += preamble;
pre_p += "Pins: gpu_a=" + gpu_a + " gpu_b=" + gpu_b + " gpu_c=" + gpu_c + " list=" + gpu_devices + "\n";
pre_p += "1) conda env " + env_name + "; torch cuda\n";
pre_p += "2) flash_attn + is_flash_attn_2_available; lm_eval; Evaluator\n";
pre_p += "3) model_path exists with weights\n";
pre_p += "4) nvidia-smi shows GPUs " + gpu_a + "," + gpu_b + "," + gpu_c + " free enough (~>30GB free each preferred)\n";
pre_p += "5) gurobipy import OK\n";
pre_p += "ok=true only if env+model+GPUs OK.\n";

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
    log("skip_model_path=true");
} else {
    let env_key = "MXMOE_MODEL_" + model_id.to_upper();
    let mp = "";
    mp += "Configure the model checkpoint path via the gitignored .env (NEVER edit tracked project_config.py).\n";
    mp += "Repo-root .env: copy from .env.example if missing; set or update the single line " + env_key + "=" + model_path + " (idempotent, keep other keys).\n";
    mp += preamble;
    mp += "Then run: python project_config.py --check " + model_id + " and verify tokenizer loads. ok=true only if the check exits 0.\n";
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
    log("skip_gate=true");
} else {
    let gp = "";
    gp += "REAL MoE gate trace on SINGLE GPU " + gpu_a + ".\n";
    gp += preamble;
    gp += "export CUDA_VISIBLE_DEVICES=" + gpu_a + "\n";
    gp += "python -m mxmoe.quant.moe_tracer --model " + model_id + " --trace_gate \\\n";
    gp += "  --dataset " + dataset + " --seqlen " + seqlen + " --nsamples " + nsamples + "\n";
    gp += "Pass: calib/gate/" + model_id + "/" + dataset + "/" + seqlen + "/moe-gate.json with access_freq.\n";
    gp += "ok=true only if pass. artifact_path=json.\n";
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
phase("Parallel-Calib");
// =============================================================================

let base_eval = ();
let c1_out = ();
let c2_out = ();

if skip_calib && (skip_baseline || !baseline_with_calib) {
    log("skip_calib and no baseline-with-calib panel");
} else {
    let jobs = [];

    if !skip_calib {
        let c1p = "";
        c1p += "MULTI-GPU calib job A on EXACTLY GPU " + gpu_a + " (do not use other GPUs).\n";
        c1p += preamble;
        c1p += "export CUDA_VISIBLE_DEVICES=" + gpu_a + "\n";
        c1p += "python -m mxmoe.quant.quant calib --model " + model_id + " --method " + qtype + " \\\n";
        c1p += "  --metric layer_out_norm --qcfg " + filter_a + " --nsamples " + nsamples + "\n";
        c1p += "Long job. ok=true if complete layer_out_norm json for " + filter_a + ".\n";
        c1p += "If a partial json exists, prefer finishing a full clean run (overwrite OK).\n";
        c1p += "artifact_path=output json. Verify nvidia-smi shows load mainly on GPU " + gpu_a + ".\n";
        jobs.push(#{
            prompt: c1p,
            label: "calib-a-gpu" + gpu_a,
            capability_mode: "all",
            output_schema: step_schema,
        });

        let c2p = "";
        c2p += "MULTI-GPU calib job B on EXACTLY GPU " + gpu_b + " (parallel with A; different scheme).\n";
        c2p += preamble;
        c2p += "export CUDA_VISIBLE_DEVICES=" + gpu_b + "\n";
        c2p += "python -m mxmoe.quant.quant calib --model " + model_id + " --method " + qtype + " \\\n";
        c2p += "  --metric layer_out_norm --qcfg " + filter_b + " --nsamples " + nsamples + "\n";
        c2p += "ok=true if complete layer_out_norm json for " + filter_b + ". artifact_path=json.\n";
        c2p += "Must NOT use GPU " + gpu_a + " or " + gpu_c + ".\n";
        jobs.push(#{
            prompt: c2p,
            label: "calib-b-gpu" + gpu_b,
            capability_mode: "all",
            output_schema: step_schema,
        });
    }

    if baseline_with_calib {
        let bp = "";
        bp += "MULTI-GPU baseline eval on EXACTLY GPU " + gpu_c + " IN PARALLEL with calib jobs.\n";
        bp += preamble;
        bp += "export CUDA_VISIBLE_DEVICES=" + gpu_c + "\n";
        bp += "model_id=" + model_id + " tasks=" + task_args + "\n";
        if paper_note != "" {
            bp += "paper_note: " + paper_note + "\n";
        }
        bp += "Run full-precision / baseline PPL (+ zero-shot if in tasks) via quant.py eval or Evaluator.\n";
        bp += "Do not wait for calib. ok=true if finite PPL>1. Record ppl and optional acc_avg.\n";
        jobs.push(#{
            prompt: bp,
            label: "eval-baseline-gpu" + gpu_c,
            capability_mode: "all",
            output_schema: eval_schema,
        });
    }

    if jobs.len() == 0 {
        log("Parallel-Calib: nothing to run");
    } else {
        log("Launching " + jobs.len().to_string() + " parallel GPU jobs (calib/baseline fan-out)");
        let pres = parallel(jobs);

        // Map results by position: [calib-a?, calib-b?, baseline?]
        let idx = 0;
        if !skip_calib {
            let r0 = pres[idx];
            idx += 1;
            if r0 == () || !r0.success || r0.output == () || r0.output.ok != true {
                let d = "calib-a failed";
                if r0 != () && r0.success && r0.output != () {
                    d = r0.output.summary + " | " + r0.output.details;
                }
                pause("infra", d);
            }
            c1_out = r0;
            log("Calib A OK: " + r0.output.summary);

            let r1 = pres[idx];
            idx += 1;
            if r1 == () || !r1.success || r1.output == () || r1.output.ok != true {
                let d = "calib-b failed";
                if r1 != () && r1.success && r1.output != () {
                    d = r1.output.summary + " | " + r1.output.details;
                }
                pause("infra", d);
            }
            c2_out = r1;
            log("Calib B OK: " + r1.output.summary);
        }
        if baseline_with_calib {
            let r2 = pres[idx];
            if r2 == () || !r2.success || r2.output == () || r2.output.ok != true {
                let d = "baseline-with-calib failed";
                if r2 != () && r2.success && r2.output != () {
                    d = r2.output.summary + " | " + r2.output.details;
                }
                pause("verification", d);
            }
            base_eval = r2;
            log("Baseline (parallel) OK: " + r2.output.summary);
        }
    }
}

// =============================================================================
phase("Align-Loss");
// =============================================================================

let alp = "";
alp += "Align calib JSON with EXPERT_QUANT_LOSS for model_id=" + model_id + ".\n";
alp += preamble;
alp += "qtype=" + qtype + " schemes " + filter_a + " and " + filter_b + "\n";
alp += "Symlink/copy or patch paths so both open. ok=true if both resolve.\n";
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
log("Align OK: " + al.output.summary);

// =============================================================================
phase("Solver");
// =============================================================================

let qconfig_hint = "qconfigs/" + filter_a + "+" + filter_b + "/" + model_id + "_" + qtype + "_Slayer_bs" + solver_batch + "_wbits" + wbits + "_r" + r_val + ".json";

if skip_solver {
    log("skip_solver=true");
} else {
    let sp = "";
    sp += "bits_solver ILP (CPU/Gurobi; GPU not required).\n";
    sp += preamble;
    sp += "copy perf/performance_table.json -> calib/perf/ if needed.\n";
    sp += "python -m mxmoe.quant.bits_solver --model " + model_id + " --qtype " + qtype + " \\\n";
    sp += "  --wbits " + wbits + " --solve_mode layer --batch " + solver_batch + " --r " + r_val + " \\\n";
    sp += "  --filter_list " + filter_a + " " + filter_b + " \\\n";
    sp += "  --trace_file calib/gate/" + model_id + "/" + dataset + "/" + seqlen + "/moe-gate.json \\\n";
    sp += "  --perf_file calib/perf/performance_table.json\n";
    sp += "ok=true if qconfig json exists. artifact_path=qconfig.\n";
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

// =============================================================================
phase("Parallel-Eval");
// =============================================================================

let mx_eval = ();
let uni_eval = ();

// If baseline was not run with calib, run it now (possibly in parallel with mxmoe)
let need_baseline_now = !skip_baseline && !baseline_with_calib;
let need_mx = !skip_mxmoe_eval;
let need_uni = run_uniform_w4a4;

if !need_baseline_now && !need_mx && !need_uni {
    log("Parallel-Eval: nothing to run");
} else {
    let ejobs = [];

    if need_baseline_now {
        let bp = "";
        bp += "Baseline eval on GPU " + gpu_a + ".\n";
        bp += preamble;
        bp += "export CUDA_VISIBLE_DEVICES=" + gpu_a + "\n";
        bp += "tasks=" + task_args + " model=" + model_id + "\n";
        bp += "ok=true if finite PPL. Record ppl/acc_avg.\n";
        ejobs.push(#{
            prompt: bp,
            label: "eval-baseline-gpu" + gpu_a,
            capability_mode: "all",
            output_schema: eval_schema,
        });
    }

    if need_mx {
        let pin = gpu_b;
        if need_baseline_now {
            // baseline took gpu_a; mxmoe on gpu_b
            pin = gpu_b;
        } else {
            pin = gpu_a;
        }
        let ep2 = "";
        ep2 += "MxMoE mixed eval on EXACTLY GPU " + pin + ".\n";
        ep2 += preamble;
        ep2 += "export CUDA_VISIBLE_DEVICES=" + pin + "\n";
        ep2 += "qconfig hint: " + qconfig_hint + "\n";
        ep2 += "find qconfigs -name '*" + model_id + "*wbits" + wbits + "*' ! -name '*tile*'\n";
        ep2 += "python -m mxmoe.quant.quant eval --model " + model_id + " --method " + qtype + " \\\n";
        ep2 += "  --qconfig <resolved> --tasks " + task_args + "\n";
        if paper_note != "" {
            ep2 += "paper_note: " + paper_note + "\n";
        }
        ep2 += "ok=true if finite PPL. artifact_path=qconfig.\n";
        ejobs.push(#{
            prompt: ep2,
            label: "eval-mxmoe-gpu" + pin,
            capability_mode: "all",
            output_schema: eval_schema,
        });
    }

    if need_uni {
        let upin = gpu_c;
        if upin == "" || upin == gpu_a {
            upin = gpu_b;
        }
        let up = "";
        up += "Optional uniform " + filter_a + " on GPU " + upin + ".\n";
        up += preamble;
        up += "export CUDA_VISIBLE_DEVICES=" + upin + "\n";
        up += "python -m mxmoe.quant.quant eval --model " + model_id + " --method " + qtype + " \\\n";
        up += "  --qstr " + filter_a + " --tasks " + task_args + "\n";
        up += "ok=true if finite PPL.\n";
        ejobs.push(#{
            prompt: up,
            label: "eval-uniform-gpu" + upin,
            capability_mode: "all",
            output_schema: eval_schema,
        });
    }

    log("Parallel-Eval launching " + ejobs.len().to_string() + " jobs");
    let eres = parallel(ejobs);
    let ei = 0;
    if need_baseline_now {
        let er = eres[ei];
        ei += 1;
        if er == () || !er.success || er.output == () || er.output.ok != true {
            let d = "baseline eval failed";
            if er != () && er.success && er.output != () {
                d = er.output.summary + " | " + er.output.details;
            }
            pause("verification", d);
        }
        base_eval = er;
        log("Baseline OK: " + er.output.summary);
    }
    if need_mx {
        let er = eres[ei];
        ei += 1;
        if er == () || !er.success || er.output == () || er.output.ok != true {
            let d = "mxmoe eval failed";
            if er != () && er.success && er.output != () {
                d = er.output.summary + " | " + er.output.details;
            }
            pause("verification", d);
        }
        mx_eval = er;
        log("MxMoE OK: " + er.output.summary);
    }
    if need_uni {
        let er = eres[ei];
        if er == () || !er.success || er.output == () || er.output.ok != true {
            log("uniform eval failed (optional)");
        } else {
            uni_eval = er;
            log("Uniform OK: " + er.output.summary);
        }
    }
}

// =============================================================================
phase("Report");
// =============================================================================

let rep = "";
rep += "Markdown accuracy report for model_id=" + model_id + " (multi-GPU run).\n";
rep += "Do NOT write docs/. Full markdown in details; ok=true.\n";
rep += "gpus a/b/c=" + gpu_a + "/" + gpu_b + "/" + gpu_c + "\n";
rep += "qconfig=" + qconfig_hint + "\n";
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
rep += "Sections: config, multi-GPU layout, artifacts, results, skips, non-goals.\n";

let repor = agent(rep, #{
    label: "report",
    capability_mode: "all",
    output_schema: step_schema,
});
let report_body = "# reproduce-acc\nmodel_id=" + model_id + "\n";
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
    summary: "MoE accuracy multi-GPU pipeline done for " + model_id,
    model_id: model_id,
    model_path: model_path,
    gpu_a: gpu_a,
    gpu_b: gpu_b,
    gpu_c: gpu_c,
    qconfig: qconfig_hint,
    baseline_ppl: base_ppl,
    mxmoe_ppl: mx_ppl,
    baseline_acc_avg: base_avg,
    mxmoe_acc_avg: mx_avg,
    run_zeroshot: run_zeroshot,
    report_scratch: report_path,
});
