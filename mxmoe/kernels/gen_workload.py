import os
import json
import argparse
import jsbeautifier
from typing import Literal
from project_config import *


MODEL_ID_TO_LAYERS = {
    "qwen2_moe": 24,
    "qwen2_moe_57b": 28,
    "ds2": 27,
    "mixtral": 32,
}

MODEL_ID_TO_TRACE_FILE = {
    **{
        k: {
            d: f"{CUR_DIR}/calib/gate/{k}/{d}/4096/moe-gate.json"
            for d in ["wiki2", "humaneval-x"]
        }
        for k in ["qwen2_moe", "qwen2_moe_57b", "ds2", "mixtral"]
    },
}


def freq_to_prob(freq: list[int]) -> list[float]:
    total = sum(freq)
    return [f / total for f in freq]

def generate_workload_from_gate_trace(trace_file: str, num_total_tokens: int, layer_id: int, save_path: str, qcfg_file=None, qstr=None):
    assert trace_file.endswith(".json") and os.path.exists(trace_file)
    assert qcfg_file is None or qstr is None

    if qcfg_file is not None:
        assert qcfg_file.endswith(".json") and os.path.exists(qcfg_file)
        with open(qcfg_file, "r") as f:
            qconfig: dict[str, list[dict]] = json.load(f)
        LT = qconfig.pop("LT", None)
    elif qstr is not None:
        qcfg = {
            "w_bits": int(qstr.split("w")[1].split("a")[0]),
            "a_bits": int(qstr.split("a")[1].split("_g")[0]),
            "gsize": int(qstr.split("_g")[1].split("_")[0]),
            "sym": "asym" not in qstr,
        }
    else:
        qcfg = {
            "w_bits": 16,
            "a_bits": 16,
            "gsize": -1,
            "sym": True,
        }

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(trace_file, "r") as f:
        trace: dict[str] = json.load(f)

    topk = trace["topk"]
    N, K = trace["NK"]
    num_shared_experts = trace["num_shared_experts"]

    # each list corresponds to a layer    
    result = {"num_tokens": num_total_tokens}

    for key, value in trace.items():
        if not key.startswith("layer-"): continue
        layer_idx = key.split("-")[1]
        if layer_id != -1 and layer_idx != str(layer_id): continue
        def get_linear_block_qcfg(exp_idx: str, linear: Literal["gate", "down"]) -> dict:
            assert linear in ["gate", "down"]
            if qcfg_file is None:
                return qcfg
            return {
                "w_bits": qconfig[layer_idx]["experts"][exp_idx][linear]["w_bits"],
                "a_bits": qconfig[layer_idx]["experts"][exp_idx][linear]["a_bits"],
                "gsize": qconfig[layer_idx]["experts"][exp_idx][linear]["w_gsize"],
                "sym": qconfig[layer_idx]["experts"][exp_idx][linear]["w_sym"],
            }

        num_experts = len(value["access_freq"])
        prob = freq_to_prob(value["access_freq"])

        shapes = {"gate_up":[], "down": []}
        for exp_idx, freq in enumerate(prob):
            shapes["gate_up"].append({"shape": [int(freq * num_total_tokens * topk), N*2, K], **get_linear_block_qcfg(str(exp_idx), "gate")})
            shapes["down"].append({"shape": [int(freq * num_total_tokens * topk), K, N], **get_linear_block_qcfg(str(exp_idx), "down")})

        # handle shared experts
        if num_shared_experts != 0:
            shapes["gate_up"].append({"shape": [int(num_total_tokens), N*2*num_shared_experts, K], **get_linear_block_qcfg(str(num_experts), "gate")})
            shapes["down"].append({"shape": [int(num_total_tokens), K, N*num_shared_experts], **get_linear_block_qcfg(str(num_experts), "down")})

        result[f"layer-{layer_idx}"] = shapes
    
    with open(save_path, "w") as f:
        options = jsbeautifier.default_options()
        options.indent_size = 2
        f.write(jsbeautifier.beautify(json.dumps(result), options))

    print(f"Save generated workloads to `{save_path}`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate workloads for groupgemm kernel.")

    parser.add_argument("--model", type=str, default="qwen2_moe", help="Model ID.")
    parser.add_argument("--dataset", type=str, default="wiki2", help="Dataset ID.")
    parser.add_argument("--length", type=int, default=128, help="Input Sequence length(Number of Tokens).")
    parser.add_argument("--scale", type=float, nargs="+", default=[1.0], help="Scale factor for the input sequence.")
    parser.add_argument("--layer", type=int, default=-1, help="Target layer ID")
    parser.add_argument("--qconfig", type=str, help="model qconfig file.")
    parser.add_argument("--qstr", type=str, help="model qconfig str, xor with qconfig.")

    parser.add_argument("--save", type=str, help="Path to save the generated workloads.")

    ####################################################
    args = parser.parse_args(
        # []
    )

    model_id = args.model
    dataset  = args.dataset
    ####################################################

    trace_file = MODEL_ID_TO_TRACE_FILE[model_id][dataset]

    print(f"Generate workloads from `{trace_file}` with scale: {args.scale}")

    for scale in args.scale:
        input_tokens = int(args.length * scale)
        generate_workload_from_gate_trace(trace_file, input_tokens, args.layer, args.save, args.qconfig, args.qstr)

