import os
import random
import pickle
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel

from project_config import *

# def get_tokenizer(model: PreTrainedModel):
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     return tokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer:PreTrainedTokenizer, model_id: str, test_only=False):
    ID2CACHE = {
        "train":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-wiki2-train.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-wiki2-train.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-wiki2-train.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-wiki2-train.pkl",
        },
        "test":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-wiki2-test.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-wiki2-test.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-wiki2-test.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-wiki2-test.pkl",
        }
    }

    print("Loading wikitext2 ...")

    testenc_cache = ID2CACHE["test"][model_id]
    if not os.path.exists(testenc_cache):
        testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')["input_ids"]
        os.makedirs(os.path.dirname(testenc_cache), exist_ok=True)
        with open(testenc_cache, "wb") as f:
            pickle.dump(testenc, f)
    else:
        with open(testenc_cache, "rb") as f:
            testenc = pickle.load(f)

    # early return
    if test_only:
        return None, testenc

    trainenc_cache = ID2CACHE["train"][model_id]
    if not os.path.exists(trainenc_cache):
        traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        os.makedirs(os.path.dirname(trainenc_cache), exist_ok=True)
        with open(trainenc_cache, "wb") as f:
            pickle.dump(trainenc, f)
    else:
        with open(trainenc_cache, "rb") as f:
            trainenc = pickle.load(f)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # tar = inp.clone()
        # tar[:, :-1] = -100
        # trainloader.append((inp, tar))
        trainloader.append(inp)
    return trainloader, testenc

def get_humaneval_x(nsamples, seed, seqlen, tokenizer:PreTrainedTokenizer, model_id: str):
    print("Loading HumanEval-X ...")

    ID2CACHE = {
        "train":{
            "ds2": f"{CUR_DIR}/data/cache/ds2-humanevalx-train.pkl",
            "mixtral": f"{CUR_DIR}/data/cache/mixtral-humanevalx-train.pkl",
            "qwen2_moe": f"{CUR_DIR}/data/cache/qwen2_moe-humanevalx-train.pkl",
            "qwen2_moe_57b": f"{CUR_DIR}/data/cache/qwen2_moe_57b-humanevalx-train.pkl",
        },
    }


    trainenc_cache = ID2CACHE["train"][model_id]
    if not os.path.exists(trainenc_cache):
        prompts = []
        for code_type in ["python", "js", "cpp", "java", "go"]:
            data = load_dataset(f"THUDM/humaneval-x", code_type)
            prompts.extend(data["test"]["prompt"])

        trainenc = tokenizer("\n\n".join(prompts), return_tensors='pt')

        os.makedirs(os.path.dirname(trainenc_cache), exist_ok=True)
        with open(trainenc_cache, "wb") as f:
            pickle.dump(trainenc, f)
    else:
        with open(trainenc_cache, "rb") as f:
            trainenc = pickle.load(f)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader, None
