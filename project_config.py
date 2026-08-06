import glob
import os
import sys

from dotenv import load_dotenv

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = f"{CUR_DIR}/calib"

# Machine-local model checkpoint paths live in <repo>/.env (gitignored; see .env.example).
# Precedence: process environment > .env. There are NO built-in defaults on purpose:
# an unconfigured model id fails loudly below instead of silently downloading from the Hub.
load_dotenv(os.path.join(CUR_DIR, ".env"), override=False)

MODEL_ID2ENV = {
    "ds2": "MXMOE_MODEL_DS2",
    "mixtral": "MXMOE_MODEL_MIXTRAL",
    "qwen2_moe": "MXMOE_MODEL_QWEN2_MOE",
    "qwen2_moe_57b": "MXMOE_MODEL_QWEN2_MOE_57B",
}


class _ModelPaths(dict):
    """ID2NAME that tells the operator how to fix an unconfigured model id."""

    def __missing__(self, model_id: str):
        env_key = MODEL_ID2ENV.get(model_id)
        if env_key is None:
            raise KeyError(
                f"Unknown model id '{model_id}'. Known ids: {sorted(MODEL_ID2ENV)}"
            )
        raise KeyError(
            f"Model '{model_id}' is not configured: set {env_key}=/path/to/checkpoint "
            f"in {CUR_DIR}/.env (see .env.example) or in the process environment, "
            f"then run `python project_config.py --check {model_id}`."
        )


ID2NAME = _ModelPaths(
    (model_id, os.path.expanduser(os.environ[env_key]))
    for model_id, env_key in MODEL_ID2ENV.items()
    if os.environ.get(env_key)
)


def check_model_paths(model_ids: list[str] | None = None) -> bool:
    """Report model path resolution. Requested ids must be set AND valid.

    Without arguments: report all known ids; unset ids are warnings, invalid
    configured paths are failures.
    """
    requested = model_ids is not None and len(model_ids) > 0
    ids = model_ids if requested else list(MODEL_ID2ENV)
    ok = True
    for model_id in ids:
        env_key = MODEL_ID2ENV.get(model_id)
        if env_key is None:
            print(f"[FAIL] {model_id}: unknown model id (known: {sorted(MODEL_ID2ENV)})")
            ok = False
            continue
        raw = os.environ.get(env_key)
        if not raw:
            level = "FAIL" if requested else "unset"
            if requested:
                ok = False
            print(f"[{level}] {model_id}: {env_key} not set (process env or {CUR_DIR}/.env)")
            continue
        path = os.path.expanduser(raw)
        has_config = os.path.isfile(os.path.join(path, "config.json"))
        has_weights = bool(
            glob.glob(os.path.join(path, "*.safetensors"))
            or glob.glob(os.path.join(path, "*.bin"))
        )
        valid = has_config and has_weights
        if not valid:
            ok = False
        print(
            f"[{'OK' if valid else 'FAIL'}] {model_id}: {path} "
            f"(config.json={has_config}, weights={has_weights})"
        )
    return ok


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--check"]
    sys.exit(0 if check_model_paths(argv or None) else 1)

EXPERT_QUANT_LOSS = {
    "rtn": {
        # "ds2": {
        #     "w8a8_g-1_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a4_g-1_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a4_g128_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a16_g-1_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-model_out_norm.json",
        #     "w4a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        #     "w3a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        #     "w2a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        # },
        **{
            k: {
                "w8a8_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W8A8_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g128_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A4_g128_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W8A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w8a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w3a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w2a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w1a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W1A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
            }
            for k in ["qwen2_moe", "mixtral", "qwen2_moe_57b", "ds2"]
        },
    },
    "gptq": {
        "qwen2_moe": {
            "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W4A16_g-1_asym-wiki2-128-4096-model_out_norm.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W4A16_g128_asym-wiki2-128-4096-model_out_norm.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W3A16_g128_asym-wiki2-128-4096-model_out_norm.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W2A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        },
    },
    "gptq-had": {
        **{
            k: {
                "w8a8_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A4_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g128_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w8a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W8A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w3a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W3A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w2a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w1a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W1A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
            }
            for k in ["qwen2_moe", "mixtral", "qwen2_moe_57b", "ds2"]
        },
    },
    "rtn-fisher": {
        "qwen2_moe": {
            # "w8a8_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g128_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
            "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-fisher.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-fisher.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-fisher.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-fisher.json",
        },
    },
    "gptq-fisher": {
        "qwen2_moe": {
            # "w8a8_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g128_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W4A16_g-1_asym-wiki2-128-4096-fisher.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W4A16_g128_asym-wiki2-128-4096-fisher.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W3A16_g128_asym-wiki2-128-4096-fisher.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W2A16_g128_asym-wiki2-128-4096-fisher.json",
        },
    },
}
