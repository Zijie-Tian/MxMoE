#!/usr/bin/env python3
"""Validate and time a uniform GPTQ compressed-tensors checkpoint in vLLM.

Supported checkpoints are padded DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B
(``qwen2_moe``). The checkpoint is weight-only (A16), group size 128,
asymmetric. Supported weight widths are 2, 4, and 8. This script
forces Humming for the quantized dense linears, shared experts, and routed
experts, disables prefix caching, and keeps tensor parallel size 1 unless
overridden.

``--activation-bits 4`` or ``8`` reuses the same packed weights, applies dynamic
per-token INT4 or INT8 activation quantization through Humming, and disables
fallback. It uses one timing engine and checks its loaded precision
configuration, without running the separate A16-only observation pass.

The default A16 mode loads two engines:

1. Eager observation engine. It wraps ``apply_humming_linear`` and
   ``HummingExpertsBase.humming_forward`` before weight load, then runs a
   short generation. Counters record real calls for prefill and decode.
2. Timing engine. It does not install those wrappers. Whole-generation wall
   time is reported separately from the engine phase intervals below.

Timing boundaries, taken from ``RequestStateStats`` after
``disable_log_stats=False``. These are request-phase intervals, not kernel
launch durations:

- ``queue_s`` = ``scheduled_ts - queued_ts`` on the engine-core monotonic clock.
- ``prefill_s`` = ``first_token_ts - scheduled_ts``. Starts at the first
  SCHEDULED event and ends when the engine-core output object for the first
  new token is created. Includes scheduling, the prefill forward, and sampling
  of that token. Excludes frontend queue wait.
- ``decode_s`` = ``last_token_ts - first_token_ts``. Covers every later decode
  step, including scheduler gaps. Zero when only one token is generated.
- ``tpot_s`` = ``decode_s / (num_generation_tokens - 1)``. Excludes the token
  produced by prefill. Same definition as vLLM
  ``FinishedRequestStats.mean_time_per_output_token``.
- ``ttft_s`` = ``RequestStateStats.first_token_latency``. Frontend ``time.time``
  from arrival to the output-processor iteration that emits the first token.
  Includes queue time and IPC. It is not ``prefill_s``.
- ``wall_generate_s`` covers the frontend generation call, including
  submission and detokenization. Batches are enqueued behind a scheduler
  barrier before execution; single requests use ``LLM.generate``.

Optional ``--quality-data`` scores held-out token windows and greedily
completes natural prompts on this same uninstrumented timing engine, after
the measured repeats. Omit it to keep the default benchmark unchanged.

Example:
    python run_vllm_gptq.py /path/to/padded-ds2-w2a16 \\
        --prompt-len 512 --batch-size 1 --output-tokens 128 \\
        --warmup 1 --repeats 3 --output /tmp/vllm_w2a16.json
    python run_vllm_gptq.py /path/to/padded-ds2-w2a16 \\
        --quality-data /path/to/quality.json --quality-output-tokens 64 \\
        --max-model-len 4097 --prompt-len 512 --output-tokens 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import median

TIMING_BOUNDARIES = {
    "clock_engine_core": (
        "time.monotonic in the engine core. queued_ts and scheduled_ts come "
        "from EngineCoreEvent. first_token_ts and last_token_ts are "
        "EngineCoreOutputs.timestamp, set when that step's output object is created."
    ),
    "clock_frontend": (
        "time.time in the LLM process. arrival_time is request arrival. "
        "first_token_latency subtracts arrival_time from the output-processor "
        "iteration timestamp."
    ),
    "prefill_s": (
        "first_token_ts - scheduled_ts. First SCHEDULED event through the "
        "engine-core timestamp of the step that emits the first new token. "
        "Not a kernel launch duration."
    ),
    "decode_s": (
        "last_token_ts - first_token_ts. First new-token output timestamp "
        "through the last. Includes every decode step and scheduler gap."
    ),
    "tpot_s": (
        "decode_s / (num_generation_tokens - 1). Excludes the prefill token. "
        "Null when fewer than 2 tokens are generated."
    ),
    "ttft_s": (
        "RequestStateStats.first_token_latency. Frontend wall clock from "
        "arrival to first token, including queue and IPC. Not equal to prefill_s."
    ),
    "queue_s": "scheduled_ts - queued_ts on the engine-core monotonic clock.",
    "wall_generate_s": (
        "time.perf_counter around frontend generation in this process, including "
        "submission and detokenization. Batches use pause/enqueue/resume/wait; "
        "single requests use LLM.generate. Not kernel latency."
    ),
}

LIMITATIONS = [
    "prefill_s, decode_s, tpot_s, and ttft_s are engine request-phase intervals, not kernel latencies.",
    "ttft_s includes queue time and IPC. prefill_s does not. Do not treat them as the same quantity.",
    "Timed decode may use CUDA graph replay. The default A16 mode records calls in a separate eager engine; A4 and A8 use a loaded-kernel precision check with fallback disabled and no extra observation engine.",
    "Requests that share a scheduler step share engine timestamps. Per-request intervals are batch-step intervals, not isolated single-sequence kernel times.",
    "Batches are submitted behind a scheduler barrier so later prompts cannot overlap earlier requests' decode. TTFT and wall time include submission/barrier overhead.",
    "A non-zero prefix-cache hit or preemption fails the run because it changes the interval meaning.",
    "No numerical quality claim is made for the repeated prompt token.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local compressed-tensors checkpoint directory")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        default="-",
        help="JSON destination. Use - for stdout.",
    )
    parser.add_argument("--observation-output-tokens", type=int, default=4)
    parser.add_argument("--smoke-output-tokens", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help=(
            "0 uses the benchmark prompt plus the longest generated length, "
            "and also the quality window and completion lengths when "
            "--quality-data is set."
        ),
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument(
        "--quantization",
        default="compressed-tensors",
        help="vLLM quantization method. A16 uses compressed-tensors, not gptq.",
    )
    parser.add_argument(
        "--expect-intermediate-size",
        type=int,
        default=None,
        help=(
            "Dense MLP width the loaded config must match. Omit to read "
            "checkpoint config.json. A positive value overrides. 0 disables the check."
        ),
    )
    parser.add_argument(
        "--expect-moe-intermediate-size",
        type=int,
        default=None,
        help=(
            "Expert width the loaded config must match. Omit to read "
            "checkpoint config.json. A positive value overrides. 0 disables the check."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Time without CUDA graphs. Default leaves vLLM graph capture on.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--quality-data",
        default=None,
        help=(
            "Optional JSON file with dataset metadata, natural completion "
            "prompts, and held-out token windows. Omit to keep the default benchmark."
        ),
    )
    parser.add_argument(
        "--quality-output-tokens",
        type=int,
        default=64,
        help="Greedy completion length for each natural prompt. Default 64.",
    )
    args = parser.parse_args()
    if args.activation_bits in (4, 8):
        args.quantization = "humming"
    if args.prompt_len < 2:
        parser.error("--prompt-len must be at least 2 so prefill and decode differ")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_tokens < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("output tokens and repeats must be positive; warmup may be 0")
    if args.observation_output_tokens < 2:
        parser.error("--observation-output-tokens must be at least 2 to include decode")
    if args.smoke_output_tokens < 1:
        parser.error("--smoke-output-tokens must be positive")
    if args.quality_output_tokens < 1:
        parser.error("--quality-output-tokens must be positive")
    if (
        args.quality_data is not None
        and args.activation_bits in (4, 8)
        and (args.batch_size != 1 or args.tensor_parallel_size != 1)
    ):
        parser.error(
            f"--activation-bits {args.activation_bits} with --quality-data requires "
            "--batch-size 1 and --tensor-parallel-size 1"
        )
    if not Path(args.model).is_dir():
        parser.error(f"model path is not a directory: {args.model}")
    if args.quality_data is not None and not Path(args.quality_data).is_file():
        parser.error(f"quality data file does not exist: {args.quality_data}")
    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_import_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root, *parts]) if parts else root


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_ready(value.item())
    return str(value)


def write_report(path: str, report: dict) -> None:
    text = json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n"
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def config_value(config, name: str):
    if hasattr(config, name):
        return getattr(config, name)
    text = getattr(config, "text_config", None)
    if text is not None and hasattr(text, name):
        return getattr(text, name)
    return None


def hf_snapshot(llm) -> dict:
    config = llm.llm_engine.model_config.hf_config
    names = (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "first_k_dense_replace",
        "decoder_sparse_step",
        "num_hidden_layers",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts",
        "num_experts_per_tok",
        "torch_dtype",
    )
    return {name: json_ready(config_value(config, name)) for name in names}

def shape_failures(llm, args) -> list[str]:
    snapshot = hf_snapshot(llm)
    failures = []
    if args.expect_intermediate_size:
        actual = snapshot.get("intermediate_size")
        if actual != args.expect_intermediate_size:
            failures.append(
                f"intermediate_size is {actual}, expected {args.expect_intermediate_size}"
            )
    if args.expect_moe_intermediate_size:
        actual = snapshot.get("moe_intermediate_size")
        if actual != args.expect_moe_intermediate_size:
            failures.append(
                "moe_intermediate_size is "
                f"{actual}, expected {args.expect_moe_intermediate_size}"
            )
    return failures


def resolve_shape_expectations(
    model_dir: str, intermediate: int | None, moe: int | None
) -> tuple[int | None, int | None, list[str]]:
    """None reads checkpoint config. A positive value overrides. 0 disables.

    A missing auto field stays None and is a failure, not a disable.
    """
    if intermediate is not None and moe is not None:
        return intermediate, moe, []
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return intermediate, moe, [f"checkpoint config.json is missing: {path}"]
    cfg = json.loads(path.read_text())
    failures: list[str] = []
    resolved = []
    for name, given in (("intermediate_size", intermediate), ("moe_intermediate_size", moe)):
        if given is not None:
            resolved.append(given)
            continue
        raw = cfg.get(name)
        if type(raw) is int and raw > 0:
            resolved.append(raw)
        else:
            failures.append(f"checkpoint config.json {name} is missing or not a positive integer")
            resolved.append(None)
    return resolved[0], resolved[1], failures


def apply_shape_expectations(args) -> list[str]:
    intermediate, moe, failures = resolve_shape_expectations(
        args.model,
        args.expect_intermediate_size,
        args.expect_moe_intermediate_size,
    )
    args.expect_intermediate_size = intermediate
    args.expect_moe_intermediate_size = moe
    return failures


def engine_lengths(args) -> tuple[int, int]:
    longest_output = max(
        args.output_tokens,
        args.observation_output_tokens,
        args.smoke_output_tokens,
    )
    benchmark = args.prompt_len + longest_output
    quality_need = int(getattr(args, "quality_required_len", 0) or 0)
    needed = max(benchmark, quality_need)
    max_model_len = args.max_model_len or needed
    max_batched = max(args.batch_size * args.prompt_len, max_model_len)
    return max_model_len, max_batched


def build_llm(args, *, enforce_eager: bool, worker_cls: str | None, worker_extension_cls: str | None):
    from vllm import LLM
    max_model_len, max_batched = engine_lengths(args)
    kwargs = {
        "model": args.model,
        "quantization": args.quantization,
        "moe_backend": "humming",
        "linear_backend": "humming",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": enforce_eager,
        "dtype": args.dtype,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_batched,
        "max_num_seqs": args.batch_size,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.activation_bits in (4, 8):
        quant_config = json.loads((Path(args.model) / "config.json").read_text())["quantization_config"]
        # Humming's CT adapter selects the Linear target, not projection regexes.
        # Retain the checkpoint's explicit attention/router/embedding exclusions.
        for group in quant_config["config_groups"].values():
            group["targets"] = ["Linear"]
        kwargs["hf_overrides"] = {"quantization_config": quant_config}
    if worker_cls is not None:
        kwargs["worker_cls"] = worker_cls
    if worker_extension_cls is not None:
        kwargs["worker_extension_cls"] = worker_extension_cls
    return LLM(**kwargs), kwargs


def shutdown_llm(llm) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None) if engine is not None else None
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()


def rpc_all(llm, method: str, **kwargs) -> list:
    results = llm.collective_rpc(method, kwargs=kwargs)
    if not isinstance(results, list):
        return [results]
    return results


def merged_failures(reports: list[dict]) -> list[str]:
    failures = []
    for index, report in enumerate(reports):
        for failure in report.get("failures", []):
            failures.append(f"worker {index}: {failure}")
    return failures


def choose_token_id(tokenizer) -> tuple[int, str]:
    for text in ("a", "the"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        if ids and len(ids) == 1:
            return int(ids[0]), text
    return 1, "fallback_id_1"


def make_prompts(token_id: int, prompt_len: int, batch_size: int) -> list[dict]:
    token_ids = [token_id] * prompt_len
    return [{"prompt_token_ids": list(token_ids)} for _ in range(batch_size)]


def sampling(max_tokens: int, *, with_logprobs: bool):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        logprobs=1 if with_logprobs else None,
    )


def numbers_are_finite(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(numbers_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numbers_are_finite(item) for item in value)
    logprob = getattr(value, "logprob", None)
    if logprob is not None:
        return math.isfinite(float(logprob))
    return True


def phase_metrics(output, expected_prompt: int, expected_output: int) -> tuple[dict, list[str]]:
    failures = []
    metrics = output.metrics
    completion = output.outputs[0] if output.outputs else None
    token_ids = [] if completion is None else [int(token) for token in completion.token_ids]
    record = {
        "request_id": output.request_id,
        "finished": bool(output.finished),
        "finish_reason": None if completion is None else completion.finish_reason,
        "prompt_len": None if output.prompt_token_ids is None else len(output.prompt_token_ids),
        "output_len": len(token_ids),
        "text": None if completion is None else completion.text,
        "token_ids": token_ids,
        "num_cached_tokens": output.num_cached_tokens,
        "cumulative_logprob": None if completion is None else completion.cumulative_logprob,
        "logprobs_finite": None
        if completion is None or completion.logprobs is None
        else numbers_are_finite(completion.logprobs),
    }
    if not output.finished or completion is None or not token_ids:
        failures.append(f"{output.request_id} returned no finished token sequence")
    if output.prompt_token_ids is not None and len(output.prompt_token_ids) != expected_prompt:
        failures.append(
            f"{output.request_id} prompt length is {len(output.prompt_token_ids)}, "
            f"expected {expected_prompt}"
        )
    if len(token_ids) != expected_output:
        failures.append(
            f"{output.request_id} generated {len(token_ids)} tokens, expected {expected_output}"
        )
    if completion is not None and completion.finish_reason not in {None, "length"}:
        failures.append(
            f"{output.request_id} finish_reason is {completion.finish_reason}, expected length"
        )
    if (output.num_cached_tokens or 0) > 0:
        failures.append(
            f"{output.request_id} reused {output.num_cached_tokens} prefix-cache tokens"
        )
    if completion is not None and completion.logprobs is not None and not record["logprobs_finite"]:
        failures.append(f"{output.request_id} logprobs are not finite")
    if (
        completion is not None
        and completion.cumulative_logprob is not None
        and not math.isfinite(float(completion.cumulative_logprob))
    ):
        failures.append(f"{output.request_id} cumulative_logprob is not finite")
    if metrics is None:
        failures.append(f"{output.request_id} has no RequestStateStats; disable_log_stats was not honored")
        record["metrics"] = None
        return record, failures

    generated = int(metrics.num_generation_tokens)
    prefill_s = metrics.first_token_ts - metrics.scheduled_ts
    decode_s = metrics.last_token_ts - metrics.first_token_ts
    queue_s = metrics.scheduled_ts - metrics.queued_ts
    tpot_s = decode_s / (generated - 1) if generated > 1 else None
    timestamps = {
        "arrival_time": metrics.arrival_time,
        "queued_ts": metrics.queued_ts,
        "scheduled_ts": metrics.scheduled_ts,
        "first_token_ts": metrics.first_token_ts,
        "last_token_ts": metrics.last_token_ts,
    }
    record["metrics"] = {
        "ttft_s": metrics.first_token_latency,
        "queue_s": queue_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tpot_s": tpot_s,
        "num_generation_tokens": generated,
        "num_preemptions": int(metrics.num_preemptions),
        "is_corrupted": bool(metrics.is_corrupted),
        "timestamps": timestamps,
    }
    if generated != expected_output:
        failures.append(
            f"{output.request_id} metrics counted {generated} tokens, expected {expected_output}"
        )
    if metrics.num_preemptions:
        failures.append(f"{output.request_id} was preempted {metrics.num_preemptions} times")
    if metrics.is_corrupted:
        failures.append(f"{output.request_id} logits contained non-finite values")
    for name, stamp in timestamps.items():
        if name == "arrival_time":
            continue
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp <= 0:
            failures.append(f"{output.request_id} {name} is unset or non-finite ({stamp})")
    if not math.isfinite(metrics.first_token_latency) or metrics.first_token_latency < 0:
        failures.append(f"{output.request_id} ttft_s is not finite")
    if expected_output > 1 and (tpot_s is None or not math.isfinite(tpot_s) or tpot_s < 0):
        failures.append(f"{output.request_id} tpot_s is not a finite decode interval")
    return record, failures


def run_generate(llm, prompts, max_tokens: int, *, with_logprobs: bool) -> tuple[list, float]:
    started = time.perf_counter()
    params = sampling(max_tokens, with_logprobs=with_logprobs)
    if len(prompts) == 1:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    else:
        # The background engine can otherwise start decoding the first request
        # while the frontend is still submitting the rest of the same batch.
        core = llm.llm_engine.engine_core
        core.call_utility("pause_scheduler", "keep", False)
        try:
            llm.enqueue(prompts, params, use_tqdm=False)
        finally:
            core.call_utility("resume_scheduler")
        outputs = llm.wait_for_completion(use_tqdm=False)
        first_tokens = {output.metrics.first_token_ts for output in outputs}
        if len(first_tokens) != 1:
            raise RuntimeError("Batch prefill split across steps; phase timings are not comparable")
    return outputs, time.perf_counter() - started


def summarize_requests(records: list[dict]) -> dict:
    def values(name: str) -> list[float]:
        found = []
        for record in records:
            metrics = record.get("metrics") or {}
            value = metrics.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                found.append(float(value))
        return found

    summary = {}
    for name in ("ttft_s", "queue_s", "prefill_s", "decode_s", "tpot_s"):
        found = values(name)
        summary[name] = {
            "median": None if not found else median(found),
            "mean": None if not found else sum(found) / len(found),
            "min": None if not found else min(found),
            "max": None if not found else max(found),
        }
    return summary


def sample_from(record: dict | None) -> dict | None:
    if record is None:
        return None
    token_ids = record.get("token_ids") or []
    return {
        "text": record.get("text"),
        "token_ids": token_ids,
        "output_len": record.get("output_len"),
        "finish_reason": record.get("finish_reason"),
        "logprobs_finite": record.get("logprobs_finite"),
        "cumulative_logprob": record.get("cumulative_logprob"),
    }


def new_report(args) -> dict:
    max_model_len, max_batched = engine_lengths(args)
    report = {
        "ok": False,
        "schema_version": 2,
        "model": args.model,
        "timing_boundaries": TIMING_BOUNDARIES,
        "limitations": LIMITATIONS,
        "requested": {
            "prompt_len": args.prompt_len,
            "batch_size": args.batch_size,
            "output_tokens": args.output_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "observation_output_tokens": args.observation_output_tokens,
            "smoke_output_tokens": min(args.smoke_output_tokens, args.output_tokens),
            "quantization": args.quantization,
            "moe_backend": "humming",
            "linear_backend": "humming",
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_prefix_caching": False,
            "batch_submission": "scheduler_barrier" if args.batch_size > 1 else "generate",
            "expect_intermediate_size": args.expect_intermediate_size,
            "expect_moe_intermediate_size": args.expect_moe_intermediate_size,
            "weight_bits": args.weight_bits,
            "group_size": args.group_size,
            "asymmetric": True,
            "activation": f"A{args.activation_bits}",
            "enforce_eager_timing": args.enforce_eager,
            "input_quantization": {
                "dtype": f"int{args.activation_bits}",
                "group_size": 0,
                "allow_fallback": False,
            }
            if args.activation_bits in (4, 8) else None,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_batched,
            "max_num_seqs": args.batch_size,
        },
        "failures": [],
    }
    if getattr(args, "quality_data", None):
        report["requested"]["quality_data"] = args.quality_data
        report["requested"]["quality_output_tokens"] = args.quality_output_tokens
        required = getattr(args, "quality_required_len", None)
        if required:
            report["requested"]["quality_required_max_model_len"] = required
    return report


def observe(args, report: dict) -> tuple[int, str] | None:
    from mxmoe.vllm_gptq_audit import AUDIT_WORKER

    llm, kwargs = build_llm(
        args,
        enforce_eager=True,
        worker_cls=AUDIT_WORKER,
        worker_extension_cls=None,
    )
    report["observation_llm_kwargs"] = {
        key: value for key, value in kwargs.items() if key != "model"
    }
    report["observation_llm_kwargs"]["model"] = kwargs["model"]
    try:
        report["hf_config"] = hf_snapshot(llm)
        failures = shape_failures(llm, args)
        static_reports = rpc_all(
            llm,
            "gptq_audit_inspect",
            weight_bits=args.weight_bits,
            group_size=args.group_size,
        )
        report["static_selection_observation"] = static_reports
        failures.extend(merged_failures(static_reports))
        report["failures"].extend(failures)
        if failures:
            return None
        token_id, token_source = choose_token_id(llm.get_tokenizer())
        report["prompt_token"] = {"id": token_id, "source": token_source}
        prompts = make_prompts(token_id, args.prompt_len, args.batch_size)
        rpc_all(
            llm,
            "gptq_audit_begin",
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
        )
        outputs, wall_s = run_generate(
            llm,
            prompts,
            args.observation_output_tokens,
            with_logprobs=True,
        )
        observed = rpc_all(llm, "gptq_audit_end")
        report["observed_calls"] = observed
        records = []
        for output in outputs:
            record, record_failures = phase_metrics(
                output, args.prompt_len, args.observation_output_tokens
            )
            records.append(record)
            report["failures"].extend(record_failures)
        report["observation_generate"] = {
            "wall_generate_s": wall_s,
            "requests": records,
            "note": (
                "Eager generate used only to observe Humming calls. "
                "Its wall time is not the performance result."
            ),
        }
        report["observation_sample"] = sample_from(records[0] if records else None)
        report["failures"].extend(merged_failures(observed))
        return token_id, token_source
    finally:
        try:
            rpc_all(llm, "gptq_audit_uninstall")
        except Exception:
            traceback.print_exc()
        shutdown_llm(llm)


def time_engine(args, report: dict, token_id: int | None) -> None:
    from mxmoe.vllm_gptq_audit import WORKER_EXTENSION

    llm, kwargs = build_llm(
        args,
        enforce_eager=args.enforce_eager,
        worker_cls=None,
        worker_extension_cls=WORKER_EXTENSION,
    )
    report["timing_llm_kwargs"] = dict(kwargs)
    try:
        shape_errors = shape_failures(llm, args)
        report["failures"].extend(shape_errors)
        if args.activation_bits in (4, 8):
            static_reports = rpc_all(
                llm, "gptq_audit_inspect_a4",
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
            )
        else:
            static_reports = rpc_all(
                llm, "gptq_audit_inspect", weight_bits=args.weight_bits,
                group_size=args.group_size,
            )
        report["static_selection_timing"] = static_reports
        static_failures = merged_failures(static_reports)
        report["failures"].extend(static_failures)
        if shape_errors or static_failures:
            if shape_errors and static_failures:
                reason = (
                    "loaded config shape does not match expectations; "
                    f"timing engine did not select the Humming W{args.weight_bits}A{args.activation_bits} path"
                )
            elif shape_errors:
                reason = "loaded config shape does not match expectations"
            else:
                reason = (
                    f"timing engine did not select the Humming W{args.weight_bits}A{args.activation_bits} path"
                )
            report["timing_skipped"] = reason
            _quality_skipped(args, report, "quality evaluation was not run because " + reason)
            return
        if token_id is None:
            token_id, token_source = choose_token_id(llm.get_tokenizer())
            report["prompt_token"] = {"token_id": token_id, "source": token_source}
        prompts = make_prompts(token_id, args.prompt_len, args.batch_size)
        smoke_tokens = min(args.smoke_output_tokens, args.output_tokens)
        smoke_outputs, smoke_wall = run_generate(
            llm, prompts, smoke_tokens, with_logprobs=True
        )
        smoke_records = []
        for output in smoke_outputs:
            record, failures = phase_metrics(output, args.prompt_len, smoke_tokens)
            smoke_records.append(record)
            report["failures"].extend(failures)
        report["smoke"] = {
            "output_tokens": smoke_tokens,
            "wall_generate_s": smoke_wall,
            "requests": smoke_records,
            "note": "Smoke checks a real token sequence. It is not a timed repeat.",
        }
        report["sample"] = sample_from(smoke_records[0] if smoke_records else None)

        warmup_walls = []
        for _ in range(args.warmup):
            _, wall_s = run_generate(
                llm, prompts, args.output_tokens, with_logprobs=False
            )
            warmup_walls.append(wall_s)
        repeats = []
        for _ in range(args.repeats):
            outputs, wall_s = run_generate(
                llm, prompts, args.output_tokens, with_logprobs=False
            )
            records = []
            for output in outputs:
                record, failures = phase_metrics(output, args.prompt_len, args.output_tokens)
                records.append(record)
                report["failures"].extend(failures)
            repeats.append(
                {
                    "wall_generate_s": wall_s,
                    "requests": records,
                    "batch_prefill_s": _batch_max(records, "prefill_s"),
                    "batch_decode_s": _batch_max(records, "decode_s"),
                    "batch_tpot_s": _batch_max(records, "tpot_s"),
                    "batch_ttft_s": _batch_max(records, "ttft_s"),
                }
            )
        report["measurements"] = {
            "warmup_wall_generate_s": warmup_walls,
            "repeats": repeats,
            "summary_over_repeats": {
                "wall_generate_s_median": _median_defined(
                    [item["wall_generate_s"] for item in repeats]
                ),
                "prefill_s_median": _median_defined(
                    [item["batch_prefill_s"] for item in repeats]
                ),
                "decode_s_median": _median_defined(
                    [item["batch_decode_s"] for item in repeats]
                ),
                "tpot_s_median": _median_defined(
                    [item["batch_tpot_s"] for item in repeats]
                ),
                "ttft_s_median": _median_defined(
                    [item["batch_ttft_s"] for item in repeats]
                ),
            },
            "per_request_summary": summarize_requests(
                [record for item in repeats for record in item["requests"]]
            ),
        }
        _record_quality(args, report, llm)
    finally:
        shutdown_llm(llm)


def _batch_max(records: list[dict], name: str) -> float | None:
    found = []
    for record in records:
        metrics = record.get("metrics") or {}
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            found.append(float(value))
    if not found:
        return None
    return max(found)


def _median_defined(values: list[float | None]) -> float | None:
    found = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not found:
        return None
    return float(median(found))


class QualityDataError(ValueError):
    """Quality input or score is unusable. The message is safe for the JSON report."""


@dataclass(frozen=True)
class QualityData:
    dataset: dict
    prompts: list[str]
    token_windows: list[list[int]]
    source: str


_QUALITY_PROTOCOL = {
    "name": "held-out window perplexity",
    "scoring": "prompt_logprobs=0, max_tokens=1, temperature=0, ignore_eos; one window per request",
    "scored_positions": "1..len(window)-1",
    "excluded_positions": "window index 0 only",
    "nll": "-log p(target | preceding window tokens)",
    "aggregation": "exp(sum(NLL) / predicted tokens), weighted by predicted-token count",
    "generation": "greedy temperature=0, max_tokens=output_tokens; one raw completion per prompt",
    "prompt_encoding": "tokenizer.encode(text, add_special_tokens=True)",
    "separated_from_benchmark_timing": True,
}
_QUALITY_LIMITATION = (
    "Held-out window perplexity from teacher-forced prompt logprobs. "
    "Each window's first token is excluded. Not a paper reproduction. "
    "Finite logprobs are an integrity check, not a quality threshold."
)


def load_quality_data(path: str | Path) -> QualityData:
    """Load and validate the quality JSON contract. Keeps the parsed lists."""
    source = str(path)
    file = Path(path)
    if not file.is_file():
        raise QualityDataError(f"quality data file does not exist: {source}")
    try:
        payload = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise QualityDataError(f"quality data is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualityDataError(f"quality data must be a JSON object: {source}")
    dataset = payload.get("dataset")
    prompts = payload.get("prompts")
    windows = payload.get("token_windows")
    if not isinstance(dataset, dict):
        raise QualityDataError(f"{source} dataset must be an object")
    for name in ("name", "split"):
        if not isinstance(dataset.get(name), str) or not dataset.get(name):
            raise QualityDataError(f"{source} dataset.{name} must be a non-empty string")
    length = dataset.get("sequence_length")
    if type(length) is not int or length < 2:
        raise QualityDataError(f"{source} dataset.sequence_length must be an integer >= 2")
    if not isinstance(prompts, list) or not prompts:
        raise QualityDataError(f"{source} prompts must be a non-empty list")
    for index, text in enumerate(prompts):
        if not isinstance(text, str) or text == "":
            raise QualityDataError(f"{source} prompts[{index}] must be a non-empty string")
    if not isinstance(windows, list) or not windows:
        raise QualityDataError(f"{source} token_windows must be a non-empty list")
    declared = dataset.get("windows")
    if declared is not None and declared != len(windows):
        raise QualityDataError(
            f"{source} dataset.windows is {declared}, but token_windows has {len(windows)}"
        )
    for index, window in enumerate(windows):
        where = f"{source} token_windows[{index}]"
        if not isinstance(window, list):
            raise QualityDataError(f"{where} must be a list")
        if len(window) != length:
            raise QualityDataError(
                f"{where} has {len(window)} tokens, expected sequence_length {length}"
            )
        for token in window:
            if type(token) is not int:
                raise QualityDataError(f"{where} is not an integer token id")
            if token < 0:
                raise QualityDataError(f"{where} is negative")
    return QualityData(dataset, prompts, windows, source)


def _exp_ppl(mean_nll: float, where: str) -> float:
    if not math.isfinite(mean_nll):
        raise QualityDataError(f"{where} is non-finite")
    try:
        ppl = math.exp(mean_nll)
    except OverflowError as exc:
        raise QualityDataError(f"{where} is non-finite") from exc
    if not math.isfinite(ppl):
        raise QualityDataError(f"{where} is non-finite")
    return ppl


def score_window_logprobs(
    token_ids: list[int],
    prompt_logprobs,
    *,
    index: int = 0,
) -> dict[str, float | int]:
    """NLL for positions 1..len-1. Fail on missing, truncated, or non-finite data."""
    if len(token_ids) < 2:
        raise QualityDataError(
            f"window {index} has {len(token_ids)} tokens; at least 2 are required"
        )
    if prompt_logprobs is None:
        raise QualityDataError(
            f"window {index} prompt logprobs are missing; refusing to skip this window"
        )
    try:
        covered = len(prompt_logprobs)
    except TypeError as exc:
        raise QualityDataError(f"window {index} prompt logprobs are not a sequence") from exc
    if covered != len(token_ids):
        raise QualityDataError(
            f"window {index} prompt logprobs cover {covered} positions, "
            f"expected {len(token_ids)}; refusing to score a truncated window"
        )
    nlls = []
    for position in range(1, len(token_ids)):
        where = f"window {index} position {position}"
        token_id = token_ids[position]
        entry = prompt_logprobs[position]
        if not isinstance(entry, dict) or not entry:
            raise QualityDataError(
                f"{where} is missing prompt logprobs for target token {token_id}"
            )
        if token_id not in entry:
            raise QualityDataError(
                f"{where} is missing target token {token_id}; prompt logprob keys are {list(entry)}"
            )
        value = entry[token_id]
        number = value if type(value) in (int, float) else getattr(value, "logprob", None)
        if type(number) not in (int, float) or not math.isfinite(number):
            raise QualityDataError(f"{where} logprob for target token {token_id} is non-finite")
        nlls.append(-float(number))
    nll_sum = math.fsum(nlls)
    if not math.isfinite(nll_sum):
        raise QualityDataError(f"window {index} NLL sum is non-finite")
    predicted = len(token_ids) - 1
    mean_nll = nll_sum / predicted
    return {
        "index": index,
        "tokens": len(token_ids),
        "predicted_tokens": predicted,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": _exp_ppl(mean_nll, f"window {index} perplexity"),
    }


def aggregate_window_nlls(windows: list[dict]) -> dict[str, float | int]:
    """Weight each window by its predicted-token count. Do not average PPLs."""
    if not windows:
        raise QualityDataError("no windows to aggregate")
    predicted = 0
    nll_sum = 0.0
    for window in windows:
        count = window.get("predicted_tokens")
        total = window.get("nll_sum")
        index = window.get("index", "?")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise QualityDataError(
                f"window {index} predicted token count is {count!r}, expected a positive integer"
            )
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(float(total))
        ):
            raise QualityDataError(f"window {index} NLL sum is non-finite")
        predicted += count
        nll_sum += float(total)
    mean_nll = nll_sum / predicted
    return {
        "nll_sum": nll_sum,
        "predicted_tokens": predicted,
        "mean_nll": mean_nll,
        "ppl": _exp_ppl(mean_nll, "held-out window perplexity"),
    }


def length_failure(
    max_model_len: int,
    *,
    window_tokens: int,
    prompt_tokens: int,
    output_tokens: int,
) -> str | None:
    """Return a refusal message when an explicit context length would truncate."""
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        return (
            f"max_model_len must be a positive integer to score quality data, "
            f"got {max_model_len!r}. Refusing to truncate."
        )
    window_need = window_tokens + 1
    prompt_need = prompt_tokens + output_tokens
    need = max(window_need, prompt_need)
    if max_model_len >= need:
        return None
    return (
        f"max_model_len {max_model_len} is smaller than {need}: "
        f"largest token window is {window_tokens} and teacher-forced scoring "
        f"generates 1 token (need {window_need}); longest natural prompt is "
        f"{prompt_tokens} tokens and quality generation requests {output_tokens} "
        f"tokens (need {prompt_need}). Refusing to truncate."
    )


def _encode_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    encoded = []
    for index, text in enumerate(prompts):
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            raise QualityDataError(f"prompts[{index}] encodes to no tokens")
        encoded.append(ids)
    return encoded


def _require_echo(output, expected, where: str) -> None:
    echoed = output.prompt_token_ids
    if (
        echoed is None
        or len(echoed) != len(expected)
        or any(got != want for got, want in zip(echoed, expected))
    ):
        got = 0 if echoed is None else len(echoed)
        raise QualityDataError(
            f"{where} prompt was truncated or rewritten "
            f"({got} tokens returned, {len(expected)} submitted); refusing to score"
        )


def _one_output(llm, prompt: dict, params):
    outputs = llm.generate([prompt], params, use_tqdm=False)
    if not outputs or len(outputs) != 1:
        raise QualityDataError(
            f"expected one generate result, got {0 if not outputs else len(outputs)}"
        )
    return outputs[0]


def evaluate_quality(llm, data: QualityData, *, output_tokens: int = 64) -> dict:
    """Score windows and greedily complete prompts on an existing LLM."""
    if type(output_tokens) is not int or output_tokens < 1:
        raise QualityDataError("quality output tokens must be a positive integer")
    prompt_ids = _encode_prompts(llm.get_tokenizer(), data.prompts)
    failure = length_failure(
        llm.llm_engine.model_config.max_model_len,
        window_tokens=max(len(window) for window in data.token_windows),
        prompt_tokens=max(len(ids) for ids in prompt_ids),
        output_tokens=output_tokens,
    )
    if failure:
        raise QualityDataError(failure)
    from vllm import SamplingParams

    generations = []
    params = SamplingParams(temperature=0.0, max_tokens=output_tokens)
    for index, (text, token_ids) in enumerate(zip(data.prompts, prompt_ids)):
        output = _one_output(llm, {"prompt": text, "prompt_token_ids": token_ids}, params)
        _require_echo(output, token_ids, f"prompt {index}")
        if not output.outputs:
            raise QualityDataError(f"prompt {index} returned no completion")
        completion = output.outputs[0]
        output_ids = completion.token_ids
        finish = completion.finish_reason
        n_out = len(output_ids)
        if n_out == 0:
            raise QualityDataError(f"prompt {index} generated no tokens")
        if n_out > output_tokens:
            raise QualityDataError(
                f"prompt {index} generated {n_out} tokens, expected at most {output_tokens}"
            )
        if finish == "length" and n_out != output_tokens:
            raise QualityDataError(
                f"prompt {index} length-stopped at {n_out} tokens, expected {output_tokens}; "
                "refusing a truncated completion"
            )
        if finish not in {None, "length", "stop"}:
            raise QualityDataError(f"prompt {index} finish_reason is {finish}")
        generations.append(
            {
                "prompt": text,
                "prompt_token_ids": token_ids,
                "output_text": completion.text,
                "output_token_ids": output_ids,
                "finish_reason": finish,
            }
        )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, ignore_eos=True)
    scores = []
    for index, window in enumerate(data.token_windows):
        output = _one_output(llm, {"prompt_token_ids": window}, params)
        _require_echo(output, window, f"window {index}")
        n_out = len(output.outputs[0].token_ids) if output.outputs else 0
        if n_out != 1:
            raise QualityDataError(
                f"window {index} generated {n_out} tokens, expected 1; "
                "max_model_len may be too small to score the full window"
            )
        scores.append(score_window_logprobs(window, output.prompt_logprobs, index=index))
    perplexity = aggregate_window_nlls(scores)
    return {
        "ran": True,
        "kind": "held_out_window_perplexity",
        "paper_reproduction": False,
        "quality_threshold": None,
        "limitation": _QUALITY_LIMITATION,
        "dataset": data.dataset,
        "protocol": _QUALITY_PROTOCOL,
        "counts": {
            "windows": len(data.token_windows),
            "prompts": len(data.prompts),
            "generations": len(generations),
            "window_tokens": sum(len(window) for window in data.token_windows),
            "predicted_tokens": perplexity["predicted_tokens"],
            "excluded_first_tokens": len(data.token_windows),
            "quality_output_tokens": output_tokens,
        },
        "perplexity": perplexity,
        "windows": scores,
        "generations": generations,
    }


def prepare_quality(args) -> None:
    args.quality_payload = None
    args.quality_required_len = 0
    if not args.quality_data:
        return
    data = load_quality_data(args.quality_data)
    try:
        from vllm.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    except Exception as exc:
        raise QualityDataError(
            f"cannot load the checkpoint tokenizer to size max_model_len: {exc}"
        ) from exc
    lengths = [len(ids) for ids in _encode_prompts(tokenizer, data.prompts)]
    window_tokens = max(len(window) for window in data.token_windows)
    prompt_tokens = max(lengths)
    if args.max_model_len:
        failure = length_failure(
            args.max_model_len,
            window_tokens=window_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=args.quality_output_tokens,
        )
        if failure:
            raise QualityDataError(failure)
    args.quality_payload = data
    args.quality_required_len = max(window_tokens + 1, prompt_tokens + args.quality_output_tokens)


def _quality_error(message: str) -> dict:
    return {
        "ran": False,
        "kind": "held_out_window_perplexity",
        "paper_reproduction": False,
        "quality_threshold": None,
        "error": message,
    }


def _quality_skipped(args, report: dict, reason: str) -> None:
    if not getattr(args, "quality_data", None):
        return
    report["failures"].append(reason)
    report["quality"] = _quality_error(reason)


def _record_quality(args, report: dict, llm) -> None:
    if not getattr(args, "quality_data", None):
        return
    data = args.quality_payload
    print(
        f"quality: scoring {len(data.token_windows)} windows and {len(data.prompts)} prompts "
        "outside measured repeats",
        file=sys.stderr,
    )
    try:
        report["quality"] = evaluate_quality(
            llm,
            data,
            output_tokens=args.quality_output_tokens,
        )
    except QualityDataError as exc:
        report["failures"].append(str(exc))
        report["quality"] = _quality_error(str(exc))
        return
    if args.activation_bits in (4, 8):
        report["quality"]["activation"] = {
            "bits": args.activation_bits,
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "allow_fallback": False,
            "requests": "one at a time",
        }


def run(args, report: dict) -> int:
    import vllm

    report["vllm_version"] = getattr(vllm, "__version__", "unknown")
    if args.activation_bits in (4, 8):
        time_engine(args, report, None)
    else:
        token = observe(args, report)
        if token is None:
            report["timing_skipped"] = (
                "checkpoint shape or static Humming selection failed; timing was not started"
            )
            _quality_skipped(
                args,
                report,
                "quality evaluation was not run because the timing engine was not started",
            )
            report["ok"] = False
            return 1
        time_engine(args, report, token[0])
    report["ok"] = not report["failures"]
    return 0 if report["ok"] else 1


def main() -> int:
    args = parse_args()
    ensure_import_path()
    if args.activation_bits in (4, 8):
        os.environ["VLLM_HUMMING_INPUT_QUANT_CONFIG"] = json.dumps(
            {"dtype": f"int{args.activation_bits}", "group_size": 0, "allow_fallback": False}
        )
    exit_code = 1
    report = None
    try:
        shape_errors = apply_shape_expectations(args)
        prepare_quality(args)
        report = new_report(args)
        if shape_errors:
            report["failures"].extend(shape_errors)
            report["timing_skipped"] = "checkpoint config could not supply shape checks"
            _quality_skipped(
                args,
                report,
                "quality evaluation was not run because checkpoint config could not supply shape checks",
            )
            report["ok"] = False
            exit_code = 1
        else:
            exit_code = run(args, report)
    except QualityDataError as exc:
        if report is None:
            report = new_report(args)
        report["ok"] = False
        report["failures"].append(str(exc))
        report["error"] = str(exc)
        report["quality"] = _quality_error(str(exc))
        print(str(exc), file=sys.stderr)
        exit_code = 1
    except Exception:
        if report is None:
            report = new_report(args)
        report["ok"] = False
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
        exit_code = 1
    write_report(args.output, report)
    if args.output != "-":
        print(
            f"wrote {args.output} ok={report['ok']} failures={len(report['failures'])}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
