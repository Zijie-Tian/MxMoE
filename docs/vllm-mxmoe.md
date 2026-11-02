# 在 vLLM 中使用原生 MxMoE

本文说明 checkpoint 准备、A16 运行、逐 expert/projection 的 activation 策略、验证方法和常见错误。基本入口见 [AGENTS.md](../AGENTS.md)，环境安装只以 [统一环境指南](../doc/environment.md) 为准。

## 1. 功能范围

`moe_backend="mxmoe"` 显式选择原生 MxMoE routed-expert 算子；`auto` 的后端优先级不变。实现与专用部署检查位于 [`mxmoe/vllm/`](../mxmoe/vllm/)，本地 vLLM 子模块只保留注册、配置和 checkpoint 适配入口。

| 项目 | 要求 / 行为 |
|---|---|
| 环境 | Conda `mxmoe`；使用仓库记录的本地 editable vLLM 子模块 |
| GPU | CUDA SM80 或更新架构 |
| 模型接口 | FP16，显式指定 `dtype="half"` |
| 权重格式 | `compressed-tensors`，静态、非对称、group128 的 W2/W4/W8 |
| 并行 | 单 GPU，TP/PP/DP=1，EP/EPLB 关闭 |
| 激活函数 | 标准 gated SiLU，不支持额外 clamp/scaled/SituGLU 参数 |
| 维度 | hidden size 与每个 expert intermediate size 均为 128 的倍数 |
| 其他限制 | 不支持 BF16 接口、act-order、LoRA、expert bias、runtime expert padding、router-weight-on-input |
| 默认 activation | 没有指定策略时为 A16 |

| Checkpoint 权重 | 原生支持的 activation |
|---|---|
| W2 | A4、A8、A16 |
| W4 | A4、A8、A16 |
| W8 | A8、A16；W8A4 会被明确拒绝 |

本后端负责 **routed experts**。Shared experts 和非 MoE 的 dense MLP 继续由所选 linear backend 执行，精度由 checkpoint 决定。当前导出器会量化 expert、shared expert，以及 DeepSeek 首层 dense MLP；不能因此把其余所有模块都称为 FP16。

### 格式不可混用

- `hf_float` 是 fake-quant 浮点权重，不是原生后端需要的 packed CT checkpoint。
- `gptq_pack` 是另一套 GPTQ 张量布局，不等于 CT。
- 导出命令的格式名称是 `compressed_tensors`（下划线）；vLLM 的 quantization 名称是 `compressed-tensors`（连字符）。
- Activation 策略不会重做 GPTQ、转换权重量化方案或导出不同 tensor 使用不同 W bits 的 checkpoint。

## 2. 环境、路径和模型检查

所有命令从仓库根目录执行。先按环境指南完成安装，并保留父仓库记录的 submodule 版本，不临时切换到 upstream 最新版。

把 `.env.example` 复制为本机的 `.env` 后填写：

| 本地变量 | 用途 |
|---|---|
| `MXMOE_MODEL_DS2` | 原始 DeepSeek-V2-Lite checkpoint；用于模型路径检查和可选导出 |
| `MXMOE_DS2_W4_CT` | 已导出的完整 W4 CT checkpoint 目录 |
| `MXMOE_ACTIVATION_CONFIG` | 要加载的 activation JSON；只跑 A16 时可不设置 |
| `MXMOE_EXPORT_HF_ROOT` | 可选的大容量导出根目录；需要导出时按本机存储情况设置 |

Checkpoint 和策略路径只放在 gitignored `.env` 或进程环境中，禁止提交真实机器路径、权重、缓存和凭据。进程环境优先于 `.env`。路径允许包含 `~`，示例会展开。

```bash
conda run --no-capture-output -n mxmoe python project_config.py --check ds2
```

只有退出码为 0 才继续加载权重。`--check ds2` 检查的是原始模型路径，不会替你检查新增的 exported-checkpoint/policy 路径；后续示例会分别读取对应文件。空的 `MXMOE_DS2_W4_CT` 或 `MXMOE_ACTIVATION_CONFIG` 不是有效配置。

示例使用 `CUDA_VISIBLE_DEVICES=1`。根据实际空闲 GPU 调整；该物理 GPU 在进程内对应逻辑 `cuda:0`。不要为测试终止其他用户的进程。

## 3. 准备 packed checkpoint

**已有兼容 CT checkpoint 时跳过本节。** 完整 GPTQ 导出会加载原模型、执行校准并写入大文件，不是短 smoke。

先在 `.env` 设置可用的 `MXMOE_EXPORT_HF_ROOT`，然后执行：

```bash
conda run --no-capture-output -n mxmoe python project_config.py --check ds2
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mxmoe \
  python -m mxmoe.quant.export_gptq_hf \
  --model ds2 --wbits 4 --groupsize 128 \
  --nsamples 128 --seqlen 4096 --seed 42 \
  --device-map cpu --export-format compressed_tensors --pad-dense-ffn \
  --save-name ds2-gptq-w4-g128-asym-ct-padffn
```

把导出器实际输出的目录写入 `.env` 的 `MXMOE_DS2_W4_CT`。`--save-name` 是所配置导出根目录下的名字，不保证写入仓库的 `outputs/`。

重要约束：

- 不传 `--sym`，保持原生后端要求的非对称权重。
- DeepSeek-V2-Lite 首层 dense FFN 的 intermediate size 需要对齐；`--pad-dense-ffn` 在导出时分别处理 Gate/Up 和 Down 的相关维度，不改变 routed expert 的宽度。
- CT 导出支持 W2/W4/W8，不支持 W1/W3、Hadamard/online rotation、act-order/dynamic groups 和 `--max-layers` 部分层导出。
- 导出器的 `--eval-ppl` 当前只支持 `hf_float`，不要与 `compressed_tensors` 组合。
- 导出完成后应包含配置、tokenizer、完整权重及导出元数据；不要把仅有 fake-quant state 的文件当作 CT checkpoint。

## 4. A16：使用现有 smoke / benchmark runner

该命令显式激活 `mxmoe`，读取本机 `.env` 中的 W4 CT 路径，先观察实际后端调用，再运行未插桩的短计时。

```bash
conda activate mxmoe
python project_config.py --check ds2
CKPT="$(python -c 'import os, project_config; print(os.path.expanduser(os.environ["MXMOE_DS2_W4_CT"]))')"
CUDA_VISIBLE_DEVICES=1 python run_vllm_gptq.py "$CKPT" \
  --moe-backend mxmoe --quantization compressed-tensors \
  --weight-bits 4 --activation-bits 16 --dtype half --trust-remote-code \
  --prompt-len 16 --output-tokens 4 --max-model-len 64 \
  --warmup 0 --repeats 1 --gpu-memory-utilization 0.65 \
  --output out/bench/mxmoe-a16-smoke.json
```

- Conda shell 需要先按环境指南初始化。`project_config` 的导入负责加载 `.env`，这里没有用 shell `source .env` 覆盖已有进程变量。
- `--weight-bits` 必须与 checkpoint 一致；更改该参数不会转换权重。
- 短 smoke 只验证执行链路。做性能测试时固定输入、输出、batch、预热与重复次数，不能用一次短生成宣称加速。
- 输出 JSON 中应确认 `MxMoEExperts`、prefill/decode 的原生调用，以及输入输出长度检查。

**不要给这个 runner 的 `--moe-backend mxmoe` 配 `--activation-bits 4/8`。** 该全局开关属于 Humming 的统一 activation 模式。原生逐 projection 策略使用下一节的 `LLM` API；当前 runner 没有透传这一策略文件的 CLI 开关。

## 5. 原生 mixed activation

### 5.1 准备完整策略

生产/研究评测应加载与 checkpoint 对应的有效 MxMoE qconfig。JSON 按层索引、expert 索引、`gate`/`up`/`down` 组织，索引键为字符串。每个 routed projection 必须显式包含：

| 字段 | 原生后端要求 |
|---|---|
| `w_bits` | 与 checkpoint 对应投影的位宽一致 |
| `w_gsize` | 128，与 checkpoint 一致 |
| `w_sym` | `false`，与 checkpoint 一致 |
| `a_bits` | 4、8 或 16，且符合前述 W/A 支持矩阵 |
| `a_gsize` | A4/A8 时为 `-1`，即 per-token |
| `a_sym` | A4/A8 时为 `true` |
| `a_clip` | 两个有限的 `(0,1]` 系数，分别控制负/正范围；通常为 `[1.0,1.0]` |

A16 会直接旁路低比特量化。配置读取不会根据 W bits 猜测 A bits，也不会默默为缺失的 activation 字段补 A16。只消费当前 routed experts 的条目；dense/shared 路径不由这个配置控制。

如果现有标定文件描述 symmetric/per-channel 或不同 W bits，而实际 checkpoint 是 asymmetric/group128，必须准备匹配的权重产物。**不能只改 JSON 的权重字段来绕过检查。**

以下是可选的完整 DeepSeek W4 **功能验证策略生成器**，不是标定或论文配置。先把 `MXMOE_ACTIVATION_CONFIG` 指向一个新的、位于 `out/` 等 gitignored 目录内的 JSON 文件；生成器拒绝覆盖已有文件。

```bash
conda run --no-capture-output -n mxmoe python - <<'PY'
import json
import os
from pathlib import Path
from project_config import check_model_paths

if not check_model_paths(["ds2"]):
    raise SystemExit(1)
checkpoint = Path(os.environ["MXMOE_DS2_W4_CT"]).expanduser()
policy_path = Path(os.environ["MXMOE_ACTIVATION_CONFIG"]).expanduser()
if not os.environ["MXMOE_ACTIVATION_CONFIG"].strip():
    raise ValueError("Set MXMOE_ACTIVATION_CONFIG to a new JSON path in .env")
config = json.loads((checkpoint / "config.json").read_text())
quant = config["quantization_config"]
weights = quant["config_groups"]["group_0"]["weights"]
if config["model_type"] != "deepseek_v2" or quant["quant_method"] != "compressed-tensors":
    raise ValueError("This example requires a DeepSeek-V2 CT checkpoint")
if (weights["num_bits"], weights["group_size"], weights["symmetric"]) != (4, 128, False):
    raise ValueError("This example requires asymmetric W4 group128")
if policy_path.exists():
    raise FileExistsError(f"Refusing to overwrite {policy_path}")
policy = {}
for layer in range(config["first_k_dense_replace"], config["num_hidden_layers"]):
    if layer % config.get("moe_layer_freq", 1):
        continue
    experts = {}
    for expert in range(config["n_routed_experts"]):
        activation_bits = [(4, 8, 16), (8, 16, 4), (16, 4, 8)][expert % 3]
        if layer == 2:
            activation_bits = (activation_bits[0], activation_bits[0], activation_bits[2])
        experts[str(expert)] = {
            name: {"w_bits": 4, "w_gsize": 128, "w_sym": False,
                   "a_bits": activation_bits[index], "a_gsize": -1,
                   "a_sym": True, "a_clip": [1.0, 1.0]}
            for index, name in enumerate(("gate", "up", "down"))
        }
    policy[str(layer)] = {"experts": experts}
policy_path.parent.mkdir(parents=True, exist_ok=True)
policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
print(f"Wrote verification-only activation policy: {policy_path}")
PY
```

该例覆盖 Gate A4 / Up A8 / Down A16 等组合，并在配置层索引 2 覆盖 Gate/Up 输入复用。若只想检查统一 A4，可在生成前把所有 `activation_bits` 设为 `(4,4,4)`；不要据此声称获得了校准后的质量或性能策略。

### 5.2 完整推理示例

先确认 `.env` 的 `MXMOE_DS2_W4_CT` 和 `MXMOE_ACTIVATION_CONFIG` 指向正确文件，然后执行：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mxmoe python - <<'PY'
import json
import math
import os
from pathlib import Path
from project_config import check_model_paths
from run_vllm_gptq import shutdown_llm
from vllm import LLM, SamplingParams

if not check_model_paths(["ds2"]):
    raise SystemExit(1)
checkpoint = Path(os.environ["MXMOE_DS2_W4_CT"]).expanduser()
policy = Path(os.environ["MXMOE_ACTIVATION_CONFIG"]).expanduser()
if not (checkpoint / "config.json").is_file() or not policy.is_file():
    raise ValueError("Check the exported checkpoint and activation JSON paths in .env")
llm = LLM(
    model=str(checkpoint), quantization="compressed-tensors", dtype="half",
    trust_remote_code=True, moe_backend="mxmoe", linear_backend="humming",
    kernel_config={"mxmoe_activation_config": str(policy)},
    tensor_parallel_size=1, enable_prefix_caching=False, enforce_eager=False,
    max_model_len=128, max_num_batched_tokens=128, max_num_seqs=1,
    kv_cache_memory_bytes=1024**3, seed=0,
)
try:
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    if len(completion.token_ids) != 4:
        raise RuntimeError("Smoke did not complete the requested output length")
    if not all(math.isfinite(scores[token].logprob)
               for token, scores in zip(completion.token_ids, completion.logprobs)):
        raise RuntimeError("Non-finite sampled logprobs")
    print(json.dumps({"token_ids": completion.token_ids, "text": completion.text}, ensure_ascii=False))
finally:
    shutdown_llm(llm)
PY
```

这里固定 1 GiB KV cache 是短 smoke 的预算，不是所有工作负载的推荐值。长上下文/大 batch 需要重新配置；显式 KV cache 字节数优先于 `gpu_memory_utilization`。

不要在生产服务中仅为运行示例而关闭序列化安全检查。这个推理示例不需要 `VLLM_ALLOW_INSECURE_SERIALIZATION`；若使用自定义 callable RPC 做可信本机审计，应单独评估其边界。

### 5.3 内部行为与生命周期

- Gate/Up 原始输入保持 FP16。不同方案各自生成 A4/A8，不能从已经损失精度的 A4 恢复 A8。
- 相同完整方案复用输入，但当前实现仍按 projection 执行 GEMM；混合精度还会按模式分别派发。
- Down 的低比特准备融合 SiLU×Up，并遵守现有 FP16 中间舍入语义。
- 低比特 GEMM 使用真实 S4/S8 Tensor Core；W4A8 会使用适配的 S8 布局和整数计算。Raw zero-point 修正保留 CT 权重语义。
- JSON 只在初始化/权重转换阶段读取，内容摘要参与 compile hash。修改配置后重新创建引擎，不能指望已加载引擎热更新。
- CUDA 扩展在首次权重准备或 workspace 初始化时构建，需要 `ninja` 和匹配的 Torch/CUDA 编译栈。原生 CUDA out 入口不执行编译、CPU routing-offset 回读或 GPU 分配；vLLM 热身阶段可建立/扩容 workspace，后续执行复用已准备的空间。
- Mixed 路径当前为较宽布局预留权重存储。不能假定启用 A4/A8 一定更省显存或更快。

## 6. 测试、性能和 PPL

### 回归测试

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mxmoe \
  python -m pytest -q tests/test_vllm_gptq.py
```

该现有文件覆盖 packed 权重、raw zeros、输入量化、独立/共享 Gate/Up、空 expert、尾部、非默认 stream、CUDA Graph routing replay、配置错误和兼容后端初始化。没有合适 GPU 时，GPU 用例会跳过；跳过不代表 GPU 验证通过。

### 实际调用与性能口径

模型日志应出现 `MxMoEExperts`。需要 GPU 调用证明时，在计时之外使用 profiler 观察 `mxmoe_quantize_activations`、`mxmoe_grouped_w4a4_g128_asym`、`mxmoe_grouped_w4a8_g128_asym` 和对应 A16 kernel；仅看配置字符串不足以证明执行路径。

性能评测使用真实 `LLM.generate`，固定 checkpoint、完整 activation policy、输入 token IDs、batch、输出长度、KV 预算、预热/重复次数与编译模式。记录版本和源码指纹；profile 与正式计时分开，计时必须包含量化/packing/路由，不只测预先量化后的 GEMM。

Prefill 吞吐按整批输入 token 数除以该阶段时间计算；Decode 阶段从首 token 到末 token，分子使用整批的 `output_tokens-1`。统一以 **token/s** 汇报，并说明 batch 合计还是单请求；不要混用 Decode 吞吐和包含 Prefill 的端到端输出吞吐。

### PPL 与质量边界

- 有限的生成 logprobs 仅是执行/数据完整性检查，不是 PPL。
- PPL 需要固定语料和 teacher-forced token logprobs，按总 NLL / 总有效预测 token 数聚合。
- `run_vllm_gptq.py --quality-data ...` 是既有 A16/受支持 Humming 路径的质量入口；原生 mixed policy 尚未接入该 runner 的 CLI，不能把历史 Humming PPL 当作当前 MxMoE 的结果。
- 当前 mixed 集成有真实执行与数值回归验证，但还不等于完整模型质量、输出确定性或论文性能复现。此前实测重复 greedy 生成并不总是一致，原因未在使用接口中解决。
- 保存实验数据到 `out/bench/` 等 gitignored 目录，不把逐机报告、权重或缓存加入源码提交。

## 7. 常见问题

| 现象 | 处理方式 |
|---|---|
| 找不到本地 MxMoE backend | 使用匹配的 vLLM submodule；从仓库根运行或设置正确 `PYTHONPATH`，不要换成无补丁的 upstream wheel |
| 提示 BF16 不支持 | 指定 `--dtype half` / `dtype="half"`，不要依赖 checkpoint 的默认 dtype |
| 权重 metadata 与 policy 不一致 | 核对 bits/group/symmetry 和实际 packed 产物；不要改字段伪装兼容 |
| W8A4、非对称 activation 或 grouped activation 被拒绝 | 使用支持矩阵与 `a_sym=true, a_gsize=-1`；不是所有组合都有原生 kernel |
| `--activation-bits 4/8` 不能与 native runner 组合 | 使用第 5 节的 `kernel_config.mxmoe_activation_config` API |
| 显存不足 | 降低上下文/batch/KV 预算；检查其它进程。当前 mixed 存储布局可能比 A16 对照占用更多内存 |
| 编译失败或 ABI 不兼容 | 回到统一环境指南，确认当前 `mxmoe` 的 Torch、nvcc 与扩展一致；不要通过降级依赖或切换环境规避 |
| 修改 policy 后运行结果不变 | 重建引擎；加载后的 policy 不会热更新 |
| 看到 `build` 软链接 | 它是本地缓存入口，不是应提交的源码；子模块 `.gitignore` 会排除根目录 `build`，而 `requirements/build/` 保持跟踪 |

## 8. 修改与发布约定

保持功能和部署策略集中在 `mxmoe/vllm/`；不要在 vLLM 内复制一套算子实现。更改后运行现有回归，再使用真实模型覆盖 prefill 和多 token decode。验证功能不等于证明性能或 PPL 没有变化。

需要发布父仓库与 vLLM 子模块的配套修改时，先提交并推送子模块，再在父仓库提交更新后的 gitlink 及源码/文档。必须确保父仓库引用的子模块 commit 已可从远端取得。`build`、`.env`、checkpoint、临时策略与实验结果不应进入提交；也不要为清理状态删除本地可复用构建缓存。
