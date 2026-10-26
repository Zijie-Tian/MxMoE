# MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.05799-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.05799)&nbsp;


[Haojie Duanmu](https://cat538.github.io/)<sup>1,2</sup> | [Xiuhong Li](https://scholar.google.com/citations?user=90eREm0AAAAJ&hl=en)<sup>3</sup> | [Zhihang Yuan](https://zhihang.cc/)<sup>3</sup> | [Size Zheng](https://sizezheng.github.io/)<sup>4</sup> | [Jiangfei Duan](https://jf-d.github.io/)<sup>5</sup><br> Xingcheng Zhang<sup>2</sup> | Dahua Lin<sup>2,5</sup>

<sup>1</sup>SJTU, <sup>2</sup>Shanghai AI Laboratory, <sup>3</sup>PKU, <sup>4</sup>Bytedance Seed, <sup>5</sup>CUHK.
</div>


## 🙈 TL;DR
> Can we design a quantization scheme specifically tailored to MoE models that effectively balancing model accuracy and computational efficiency?

**Insight**:

- Linear-block level: Linear blocks exhibit varying quantization sensitivity
- Expert level: Imbalanced activation frequencies $\Rightarrow$ heterogeneous computational characteristics (e.g. some experts are compute-bound while some are memory-bound).

**Approach**:

We explore the automated design of mixed-precision quantization scheme for MoE models:

1. Assign bitwidth at linear-block level.
2. Optimize bitwidth allocation (which is formulated as an ILP) by taking both model accuracy (quantization loss estimation) and computational efficiency (performance model) into consideration.
3. Generate mixed-precision GroupGEMM operator through template-based kernel generation.

![MxMoE](media/overview.png)


## 🚀 Results
![MxMoE](media/final-perf.png)


## 🛠️ Usage

1. Configure the shared environment:

    Follow the [MxMoE environment guide](doc/environment.md) for the unified
    `mxmoe` Conda environment, CUDA toolchain, and source-built extensions.
    **Accuracy and performance use the same stack**, including the local
    editable `3rdparty/vllm`; do not install a separate legacy accuracy stack.
    After setup, activate it before running the examples below:

    ```bash
    conda activate mxmoe
    ```

2. View the activation statistic of MoE models:

    ```bash
    # e.g. sample data from humaneval-x to observe qwen2_moe (in fact qwen1.5moe)
    CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.moe_tracer --model qwen2_moe --trace_gate --dataset humaneval-x
    ```

3. Calibration

    1. Get the quant loss of each linear-blocks of `<MOE_MODEL>` (e.g. `qwen2_moe`) under certain `<QUANTIZATION_CONFIG>` (e.g. `w4a4_g-1_sym`):

        ```bash
        CUDA_VISIBLE_DEVICES=0 python -m mxmoe.quant.quant calib --model qwen2_moe --method rtn --metric layer_out_norm --qcfg w4a4_g-1_sym
        ```
    2. Solve the ILP based on the quant loss and kernel profile. Quantization scheme will be saved in `qconfigs`

        ```bash
        # e.g. re-produce mxmoe w5a5
        python -m mxmoe.quant.bits_solver --model qwen2_moe --qtype gptq-had --wbits 5.0 --solve_mode layer --batch 8192 --filter_list w4a4_g-1_sym w8a8_g-1_sym
        ```

4. Accuracy Eval. You can re-produce the exp in paper by setting corresponding tasks and quantization configs.

    ```bash
    # e.g. evaluate the performance of qwen1.5_moe under RTN w4a4_g-1_sym quantization config
    CUDA_VISIBLE_DEVICES=2 python -m mxmoe.quant.quant eval --model qwen2_moe --method rtn-had --qstr w4a4_g-1_sym --tasks ppl
    ```

5. Performance (Computational Efficiency) Eval. After we get the mixed-precision scheme (Step 3), we can automatially generate corresponding GroupGEMM kernel.

    please refer to [run_mxmoe_gg.py](./run_mxmoe_gg.py).

    ```bash
    # e.g. test groupgemm in the layer-11 of qwen2_moe model (FP16):
    python run_mxmoe_gg.py --model qwen2_moe --bs 8192 --layer 11

    # e.g. test groupgemm in the layer-11 of qwen2_moe model under mixed-precision
    python run_mxmoe_gg.py --model qwen2_moe --bs 8192 --layer 11 --qconfig <QCONFIG> --tile_config <TCONFIG>
    ```

### Uniform GPTQ checkpoints for vLLM

This is separate from the mixed-precision GroupGEMM benchmark above. The
`compressed_tensors` export preserves the GPTQ codes, FP32 scales, and zero
points in the layout consumed by vLLM/Humming. `hf_float` still stores floating
weights and does not select a low-bit inference kernel.

Use the shared CUDA 12.8 `mxmoe` environment from the environment guide. Model
paths remain in `.env`; exported weights and measurement JSON stay under the
gitignored output directories:

```bash
conda run --no-capture-output -n mxmoe python project_config.py --check ds2
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  python -m mxmoe.quant.export_gptq_hf --model ds2 --wbits 2 --groupsize 128 \
  --device-map cpu --export-format compressed_tensors --pad-dense-ffn \
  --save outputs/ds2-gptq-w2-g128-asym-ct-padffn
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mxmoe \
  python run_vllm_gptq.py outputs/ds2-gptq-w2-g128-asym-ct-padffn \
  --prompt-len 512 --batch-size 1 --output-tokens 128 --warmup 3 --repeats 5 \
  --dtype half --trust-remote-code --output out/bench/ds2-w2a16-vllm.json
```

The TP=1 padding changes only the exported dense FFN width from 10944 to 11008;
the original checkpoint and expert width remain unchanged. Added quantized
weights encode zero as `q == zero_point`. This format requires static groups and
does not support Hadamard rotation, activation order, or partial-layer exports.

The runner uses a separate eager engine to observe real Humming calls, then an
uninstrumented timing engine. It checks both dense MLP/shared-expert and routed
expert W2A16 paths. Prefill, decode, TPOT, and TTFT are request-phase timings,
not isolated CUDA kernel timings; JSON includes their exact boundaries. Prefix
caching is disabled. Multi-request batches are enqueued behind a scheduler
barrier so their prefill finishes together before decode. A successful export
alone is not inference verification.

For W2A4 on Ampere, reuse the same checkpoint and add `--activation-bits 4`
with `--batch-size 1 --tensor-parallel-size 1`. Select one physical GPU per
process with `CUDA_VISIBLE_DEVICES`; independent cases may run on separate
GPUs. This mode enables dynamic per-token INT4 activations for the quantized
MLPs, leaves attention in floating point, and forbids activation fallback.
It uses one timing engine with a loaded-precision check, without the separate
A16 observation pass. The checkpoint weights are not requantized or modified.

## 👀 Limitations

⚠️ MxMoE is mainly test on RTX-4090. The current implementation cannot fully utilize the performance of hopper and later GPU architectures.

⚠️ The profile step is a little bit time consuming. We expect to upload our previously calculated quant loss statistics and kernel profile data soon, allowing you to skip this step.



## 😺 Citation
If you find MxMoE useful or relevant to your project and research, please kindly cite our paper:

```latex
@article{duanmu2025mxmoe,
  title={MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design},
  author={Duanmu, Haojie and Li, Xiuhong and Yuan, Zhihang and Zheng, Size and Duan, Jiangfei and Zhang, Xingcheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2505.05799},
  year={2025}
}
```