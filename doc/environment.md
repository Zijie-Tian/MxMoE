# MxMoE 统一环境配置

Accuracy、校准、GPTQ、导出、CUDA kernel 与 vLLM performance **全部使用同一个 Conda 环境 `mxmoe`**。统一采用 performance 所需的依赖和源码编译方式，不再保留旧 accuracy 环境，也不在 vLLM 子模块内创建 `.venv`。

## 1. 依赖基线与安装顺序

Python 依赖入口是根目录 [`requirements.txt`](../requirements.txt)，关键版本由 [`constraints.txt`](../constraints.txt) 约束；vLLM 依赖继续读取其子模块内的清单，避免复制一份后漂移。

| 组件 | 统一基线 |
|---|---|
| Python | 3.12 |
| Torch / torchvision / torchaudio | 2.11.0+cu128 / 0.26.0+cu128 / 2.11.0+cu128 |
| Transformers / Accelerate | 5.19.0 / 1.13.0 |
| CUDA Toolkit / host compiler | CUDA 12.8 / GCC、G++ 13 |
| FlashAttention | 2.8.3.post2，使用当前 Torch 从源码编译 |
| fast-hadamard-transform | 仓库子模块，从源码编译 |
| vLLM | `3rdparty/vllm`，v0.32.0 基线上的本地 editable 源码构建 |
| Rust | 以 vLLM 的 `rust-toolchain.toml` 为准，当前 1.95 |
| Triton / Humming | 3.6.0 / 0.1.17（CUDA 12 依赖） |

安装顺序：Conda 与工具链 → Python 依赖 → Hadamard、FlashAttention → vLLM 源码依赖 → editable vLLM → 验证。修改 Torch/CUDA 后必须重编原生扩展，不能复用旧 ABI 的 `.so`。

当前部署目标是 Linux x86_64 + RTX 3090（SM86），使用现有驱动可支持的 CUDA 12.8。Torch 2.11 是已发布 cu128 wheel 的最高版本，也满足本地 vLLM stable-libtorch 扩展的最低 ABI 版本；依赖声明已同步到该版本。其他 GPU 仍使用同一环境，但必须重新选择目标架构、重编扩展并验证。开发树使用动态生成的 dev 版本；复现时同时记录 `vllm.__version__` 和源码 commit，不把它当作同名 stable 发行版。

## 2. Conda、工具链和机器本地目录

以下命令从 **MxMoE 仓库根目录**执行。仅在环境不存在时创建，不删除已有的 `mxmoe`：

```bash
conda create -n mxmoe python=3.12 -y
```

随后激活；非交互任务仍应使用 `conda run --no-capture-output -n mxmoe ...`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mxmoe
export MXMOE_REPO="$PWD"
git submodule update --init --recursive
```

先在当前 shell 设置 `CUDA_HOME` 为本机 **CUDA 12.8 toolkit** 目录，`MXMOE_BUILD_ROOT` 为有足够空间的持久构建目录。这两个绝对路径不提交到仓库。已有本地子模块修改不可用 `--force` 或 reset 覆盖。若系统没有 GCC/G++ 13，可在同一环境安装 `gcc_linux-64=13 gxx_linux-64=13`，并将下方 `CC`、`CXX` 指向该环境的编译器。

```bash
: "${CUDA_HOME:?请设置 CUDA 12.8 toolkit 路径}"
: "${MXMOE_BUILD_ROOT:?请设置持久构建目录}"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-13 CXX=g++-13
export TMPDIR="$MXMOE_BUILD_ROOT/tmp"
export PIP_CACHE_DIR="$MXMOE_BUILD_ROOT/pip-cache"
export FETCHCONTENT_BASE_DIR="$MXMOE_BUILD_ROOT/vllm-deps"
export RUSTUP_HOME="$MXMOE_BUILD_ROOT/rustup"
export CARGO_HOME="$MXMOE_BUILD_ROOT/cargo"
export CARGO_TARGET_DIR="$MXMOE_BUILD_ROOT/cargo-target"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$FETCHCONTENT_BASE_DIR" "$CARGO_HOME"
export PATH="$CARGO_HOME/bin:$PATH"
export MAX_JOBS=2 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release
export TORCH_CUDA_ARCH_LIST=8.6 FLASH_ATTN_CUDA_ARCHS=86
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_USE_PRECOMPILED_RUST
nvcc --version
"$CC" --version
conda run --no-capture-output -n mxmoe python -c 'import sys; print(sys.executable)'
```

`nvcc` 必须是 12.8；不要盲从可能指向其他版本的 `cuda` 符号链接。`nvidia-smi` 中的 CUDA 版本是驱动支持上限，不是 toolkit 版本。需要支持 CUDA 12.8 的驱动，以及已安装的 [rustup](https://rustup.rs/)；安装指定 Rust 工具链：

```bash
rustup toolchain install 1.95 --profile minimal
export RUSTUP_TOOLCHAIN=1.95
rustc --version
```

以上 `export` 只在当前 shell 有效。新终端或独立构建任务必须重新设置，不能只依赖继承的 Python、CUDA 或编译器。

## 3. 安装共享 Python 依赖和原生扩展

```bash
conda run --no-capture-output -n mxmoe python -m pip install -r requirements.txt
conda run --no-capture-output -n mxmoe python -c 'import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)'
```

确认输出对应 Torch `2.11.0+cu128`、CUDA `12.8`、Transformers `5.19.0`。`requirements.txt` 不负责构建下列扩展；它们必须使用上一步已经安装的 Torch，因此禁用 build isolation。已有同版本且数值 smoke 通过的构建可复用；Torch/CUDA 变化后才使用 `--force-reinstall` 重建。

重编 FlashAttention 时必须保留 `--no-cache-dir`：`--no-binary`、`--force-reinstall` 和 `FLASH_ATTENTION_FORCE_BUILD=TRUE` 本身不能阻止 pip 复用此前从 sdist 构建的 wheel。否则更换 Torch/CUDA 后仍可能装回旧 ABI 的扩展。

CUDA 12.8 无需额外应用 CUDA 13 的 Hadamard 编译补丁；已有子模块中的兼容修正保留。以下两个扩展均使用同一 `mxmoe` 的 Python 源码构建：


```bash
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
conda run --no-capture-output -n mxmoe python -m pip install \
  --no-build-isolation --no-deps ./mxmoe/3rdparty/fast-hadamard-transform

FLASH_ATTENTION_FORCE_BUILD=TRUE \
conda run --no-capture-output -n mxmoe python -m pip install \
  --no-cache-dir --no-build-isolation --no-deps --no-binary flash-attn \
  -c constraints.txt flash-attn==2.8.3.post2
```

独立 `flash_attn` 和 vLLM 内置的 `vllm_flash_attn` 是两套扩展：其中一套成功不代表另一套已构建。

## 4. 准备 vLLM 源码依赖并安装

使用仓库记录的 vLLM 子模块版本，不临时切到 upstream 最新分支。CUDA 12.8 的 Torch 版本在根约束与子模块依赖中统一固定为 2.11；不要用 `tools/use_existing_torch.py` 删除版本约束，也不要设置 `VLLM_USE_PRECOMPILED=1`。所有原生扩展均从当前源码构建。

### 源码缓存

CMake 会下载外部源码；网络不稳定时，按下列固定版本提前下载，再用 `*_SRC_DIR` 指向本地缓存。这是源码缓存，不是预编译 wheel；首次构建仍可能下载其他依赖。以下下载块只在创建新的缓存目录时执行；已有缓存应核对版本并复用，不覆盖修改。

```bash
fetch_source() {
  local dst="$MXMOE_BUILD_ROOT/src/$1"
  test ! -e "$dst" || { printf '目录已存在，请核对并复用：%s\n' "$dst"; return 1; }
  git init "$dst" &&
  git -C "$dst" remote add origin "$2" &&
  git -C "$dst" fetch --depth 1 origin "$3" &&
  git -C "$dst" checkout --detach FETCH_HEAD
}
fetch_source triton https://github.com/triton-lang/triton.git v3.5.1
fetch_source flash-attention https://github.com/vllm-project/flash-attention.git 9cd61de38763d712bb6ce56e2a02cc2bf718c89f
fetch_source FlashMLA https://github.com/vllm-project/FlashMLA.git 0eee43b12f034b657133cf2afca6a72ebb6efccf
fetch_source DeepSelect https://github.com/vllm-project/DeepSelect.git d96d33afe1fab0d6066da49cdc91e64c2bee65ea
fetch_source FlashKDA https://github.com/vllm-project/FlashKDA.git 17a037d98da546deb4591e967cf961a43c034d8b
fetch_source DeepGEMM https://github.com/vllm-project/DeepGEMM.git 1e1842a833699298f7afc02eefb2ed168fff6938
fetch_source qutlass https://github.com/IST-DASLab/qutlass.git e74319e3405ce6d71965732880f5dc1f52371f64
fetch_source MSA https://github.com/vllm-project/MSA.git 968061ab16cb14991a5c7d4c0e8eabf531be0dcf
fetch_source tml-fa4 https://github.com/vllm-project/tml-fa4.git 75765e76a9c2c012c1f6ecd64577eb646eb4d303

git -C "$MXMOE_BUILD_ROOT/src/flash-attention" submodule update --init --depth 1 csrc/cutlass
git -C "$MXMOE_BUILD_ROOT/src/MSA" submodule update --init --depth 1 python/fmha_sm100/cutlass
git -C "$MXMOE_BUILD_ROOT/src/FlashKDA" submodule update --init --depth 1 cutlass
```

缺少 flash-attention 的 CUTLASS 会在配置阶段失败；缺少 MSA 的 CUTLASS 即使 CUDA 编译通过也可能在安装阶段失败。源码版本依据 [`cmake/external_projects`](../3rdparty/vllm/cmake/external_projects)，升级 vLLM 时同步核对，不能随意改成各项目最新版本。

```bash
export TRITON_KERNELS_SRC_DIR="$MXMOE_BUILD_ROOT/src/triton/python/triton_kernels/triton_kernels"
export VLLM_FLASH_ATTN_SRC_DIR="$MXMOE_BUILD_ROOT/src/flash-attention"
export FLASH_MLA_SRC_DIR="$MXMOE_BUILD_ROOT/src/FlashMLA"
export DEEPSELECT_SRC_DIR="$MXMOE_BUILD_ROOT/src/DeepSelect"
export FLASH_KDA_SRC_DIR="$MXMOE_BUILD_ROOT/src/FlashKDA"
export DEEPGEMM_SRC_DIR="$MXMOE_BUILD_ROOT/src/DeepGEMM"
export QUTLASS_SRC_DIR="$MXMOE_BUILD_ROOT/src/qutlass"
export FMHA_SM100_SRC_DIR="$MXMOE_BUILD_ROOT/src/MSA"
export TML_FA4_SRC_DIR="$MXMOE_BUILD_ROOT/src/tml-fa4"
```

### Ampere 的 FA3 构建边界

本地 vLLM `setup.py` 已在 `TORCH_CUDA_ARCH_LIST` 不含 9.x 时跳过请求 Hopper FA3。当前外部 flash-attention 固定版本还需保留以下 CMake 修正：空 `FA3_ARCHS` 时不定义 GPU FA3 扩展，仅保留空目标。对**新下载、尚未修正**的源码缓存应用一次；已有修正的缓存不要重复应用。

```bash
git -C "$VLLM_FLASH_ATTN_SRC_DIR" apply <<'PATCH'
diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -281,3 +281,4 @@
+    if(FA3_ARCHS)
     define_gpu_extension_target(
         _vllm_fa3_C
         DESTINATION vllm_flash_attn
@@ -323,5 +324,9 @@
     # See _vllm_fa2_C above: enforce C++20 on the target for the vLLM subproject build.
     set_target_properties(_vllm_fa3_C PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
+    else()
+        message(STATUS "FA3 is disabled because no Hopper architecture was requested.")
+        add_custom_target(_vllm_fa3_C)
+    endif()
 elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 12.0)
     message(STATUS "FA3 is disabled because CUDA version is not 12.0 or later.")
 endif ()
PATCH
```

完成依赖和源码准备后，从 MxMoE 根目录安装：

```bash
conda run --no-capture-output -n mxmoe python -m pip install \
  --no-build-isolation --no-deps --editable ./3rdparty/vllm
```

安装过程需等待 `Successfully installed vllm...`，不能在编译启动或仅生成 `.so` 时宣布完成。后续增量构建参考[子模块说明](../3rdparty/vllm/docs/contributing/incremental_build.md)；生成 preset 的 Python 必须来自 `mxmoe`，否则会把错误解释器写入 CMake 配置。

## 5. 验证：区分安装、算子执行和模型推理

### 依赖及源码来源（不加载模型）

```bash
conda run --no-capture-output -n mxmoe python -m pip check
conda run --no-capture-output -n mxmoe python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import json
import torch, transformers, accelerate, flash_attn, fast_hadamard_transform, vllm
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
import vllm.vllm_flash_attn._vllm_fa2_C
from mxmoe.quant.evaluator import Evaluator
assert torch.__version__ == '2.11.0+cu128'
assert torch.version.cuda == '12.8'
assert transformers.__version__ == '5.19.0'
assert accelerate.__version__ == '1.13.0'
assert flash_attn.__version__ == '2.8.3.post2'
assert Path(vllm.__file__).resolve().is_relative_to(Path('3rdparty/vllm').resolve())
assert json.loads(metadata.distribution('vllm').read_text('direct_url.json'))['dir_info']['editable']
print('ENV_IMPORT_PASS', torch.__version__, transformers.__version__, vllm.__version__, vllm.__file__)
PY
```

### 小规模 CUDA 数值 smoke（不加载模型、不测性能）

先用 `nvidia-smi` 确认 GPU 余量，在 shell 设置 `MXMOE_TEST_GPU` 为选定的物理 GPU 编号；不要抢占他人的推理/训练显存。

```bash
CUDA_VISIBLE_DEVICES="${MXMOE_TEST_GPU:?请选择可用 GPU}" \
conda run --no-capture-output -n mxmoe python - <<'PY'
import torch
from fast_hadamard_transform import hadamard_transform
from flash_attn import flash_attn_func
from vllm import _custom_ops as ops

torch.manual_seed(42)
x = torch.randn(4, 64, device='cuda', dtype=torch.float16)
h = torch.ones(1, 1, device='cuda')
while h.shape[0] < 64:
    h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
torch.testing.assert_close(hadamard_transform(x, scale=0.125), (x.float() @ h / 8).half(), atol=0.003, rtol=0.003)
q, k, v = [torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
torch.testing.assert_close(flash_attn_func(q, k, v, causal=True).transpose(1, 2), ref, atol=0.003, rtol=0.003)
w = torch.randn(64, device='cuda', dtype=torch.float16)
out = torch.empty_like(x)
ops.rms_norm(out, x, w, 1e-6)
ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).half() * w
torch.testing.assert_close(out, ref, atol=0.003, rtol=0.003)
torch.cuda.synchronize()
print('CUDA_NUMERIC_PASS: Hadamard, FlashAttention, vLLM RMSNorm')
PY
```

### 真实模型验证是独立验收项

模型路径只写入 gitignored `.env`。若不存在，复制 `.env.example` 后填写本节点路径；**每次加载权重前**先执行：

```bash
conda run --no-capture-output -n mxmoe python project_config.py --check qwen2_moe
```

- Accuracy：在同一个环境运行现有 GPTQ/导出/重载及 PPL 或任务评测。不能为此降级 Torch/Transformers。
- vLLM：需要显存充足的 GPU，实际加载模型并用 `LLM.generate` 覆盖 prefill 和多 token decode，检查返回 token；导入成功或单个 RMSNorm 通过不等同于此项通过。
- Performance：固定 GPU、模型、dtype、输入/输出长度、batch、warm-up、计时范围后再测吞吐/延迟。MxMoE 的 `run_mxmoe_gg.py` 是独立 GroupGEMM benchmark，不等同于 vLLM 端到端测速。

`run_vllm_gptq.py` 对具体的 packed checkpoint 分别核验实际 Humming 调用和未插桩的 prefill/decode 性能；用法见根 README。测量 JSON 保留在 `out/bench/`，不将机器本地路径、安装日志或逐机实验报告写入本文。单个算子或环境检查通过不代表某个模型已通过推理验证，也不代表 MxMoE 自定义混合精度 kernel 已接入 vLLM。
