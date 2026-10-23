""" """

import os
import json
import jsbeautifier
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from functools import partial
from torch import Tensor
from transformers import (
    PreTrainedModel,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from mxmoe.quant.data_utils import get_wikitext2, get_humaneval_x
from mxmoe.quant.moe_utils import get_expert_linears, load_hf_model, load_tokenizer, build_max_memory
from project_config import *


class MoETracer:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.model_config = model.config
        self.num_layers = self.model_config.num_hidden_layers
        self.topk = self.model_config.num_experts_per_tok
        self.K = self.model_config.hidden_size

        assert self.model_config.model_type in [
            "deepseek_v2",
            "mixtral",
            "qwen2_moe",
        ], "Unsupported model architecture."

        if self.model_config.model_type == "deepseek_v2":
            self.N = self.model_config.moe_intermediate_size
            self.num_experts = self.model_config.n_routed_experts  # (shared experts not counted)
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 4, 8, 16, 32, 48]} for _ in range(self.num_layers)]
            self.num_shared_experts = model.config.n_shared_experts
        elif self.model_config.model_type == "mixtral":
            self.N = self.model_config.intermediate_size
            self.num_experts = self.model_config.num_local_experts
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 2, 4, 6]} for _ in range(self.num_layers)]
            self.num_shared_experts = 0
        elif self.model_config.model_type == "qwen2_moe":
            self.N = self.model_config.moe_intermediate_size
            self.num_experts = self.model_config.num_experts  # (shared experts not counted)
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 4, 8, 15, 30, 45]} for _ in range(self.num_layers)]
            self.num_shared_experts = model.config.shared_expert_intermediate_size / model.config.moe_intermediate_size

        self.experts = get_expert_linears(self.model, exclude_non_moe_layer=False)

        # output of the MoE Gate of each layer: [layer_idx, num_samples * Tensor[seqlen, topk]]
        self.topk_idx: list[torch.Tensor] = []
        self.topk_weight: list[torch.Tensor] = []
        self.gate_hook_handle: list[torch.utils.hooks.RemovableHandle] = []
        # accumulate the topk_idx and topk_weight: [layer, Tensor[num_experts]]
        self.access_freq: list[torch.Tensor] = []
        self.weights_sum: list[torch.Tensor] = []

        # sample data
        self.input_samples: list[torch.Tensor] = []

    @staticmethod
    def ds2_gate_hook(m: nn.Module, inp, out, topk_idx: list, topk_weight: list):
        # topk_idx: [seqlen, topk]
        _topk_idx, _topk_weight, _ = out
        topk_idx.append(_topk_idx.cpu())
        topk_weight.append(_topk_weight.cpu())

    @staticmethod
    def mixtral_gate_hook(m: nn.Module, inp, out, top_k: int, norm_topk_prob: bool, topk_idx: list, topk_weight: list):
        if isinstance(out, tuple):
            _logits, routing_weights, selected_experts = out
        else:
            router_logits = out
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            if norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        topk_idx.append(selected_experts.cpu())
        topk_weight.append(routing_weights.cpu())

    def plug_gate_hook(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.topk_idx.append([])
            self.topk_weight.append([])

            if self.model_config.model_type == "deepseek_v2":
                if layer_idx == 0:
                    continue

                moe: nn.Module = layer.mlp
                moe_gate: nn.Module = moe.gate

                moe_gate_hook = partial(
                    MoETracer.ds2_gate_hook, topk_idx=self.topk_idx[layer_idx], topk_weight=self.topk_weight[layer_idx]
                )
                self.gate_hook_handle.append(moe_gate.register_forward_hook(moe_gate_hook))
            elif self.model_config.model_type == "mixtral":
                moe: nn.Module = layer.mlp
                moe_gate: nn.Module = moe.gate

                moe_gate_hook = partial(
                    MoETracer.mixtral_gate_hook,
                    top_k=self.topk,
                    norm_topk_prob=True,
                    topk_idx=self.topk_idx[layer_idx],
                    topk_weight=self.topk_weight[layer_idx],
                )
                self.gate_hook_handle.append(moe_gate.register_forward_hook(moe_gate_hook))
            elif self.model_config.model_type == "qwen2_moe":
                moe: Qwen2MoeSparseMoeBlock = layer.mlp
                moe_gate: nn.Module = moe.gate

                moe_gate_hook = partial(
                    MoETracer.mixtral_gate_hook,
                    top_k=self.topk,
                    norm_topk_prob=self.model_config.norm_topk_prob,
                    topk_idx=self.topk_idx[layer_idx],
                    topk_weight=self.topk_weight[layer_idx],
                )
                self.gate_hook_handle.append(moe_gate.register_forward_hook(moe_gate_hook))
            else:
                raise ValueError("Unsupported model architecture.")

    def clear_gate_hook(self):
        for handle in self.gate_hook_handle:
            handle.remove()

    @torch.no_grad()
    def trace_gate(self):
        num_samples = len(self.input_samples)
        with torch.no_grad():
            for inp in self.input_samples:
                self.model.model(inp.to(self.model.device))

        # accumulate the topk_idx and topk_weight
        for layer_idx in range(0, self.num_layers):
            if len(self.topk_idx[layer_idx]) == 0:
                self.access_freq.append(torch.zeros(self.num_experts, dtype=torch.int32))
                self.weights_sum.append(torch.zeros(self.num_experts, dtype=torch.float32))
                continue

            # sample average
            access_freq = torch.zeros(num_samples, self.num_experts, dtype=torch.float32)
            weights_sum = torch.zeros(num_samples, self.num_experts, dtype=torch.float32)
            for j in range(num_samples):
                access_freq[j].scatter_add_(
                    0, self.topk_idx[layer_idx][j].view(-1), torch.ones(self.topk_idx[layer_idx][j].numel())
                )
                weights_sum[j].scatter_add_(
                    0, self.topk_idx[layer_idx][j].view(-1), self.topk_weight[layer_idx][j].view(-1)
                )
            self.access_freq.append(access_freq.mean(dim=0).round().int())
            self.weights_sum.append(weights_sum.mean(dim=0))

            total_counts = self.access_freq[layer_idx].sum().item()
            for topk in self.percentile_stats[layer_idx]:
                topk_expert_ids = self.access_freq[layer_idx].topk(topk).indices.tolist()
                topk_counts = self.access_freq[layer_idx][topk_expert_ids].tolist()
                self.percentile_stats[layer_idx][topk] = {
                    "topk_ids": topk_expert_ids,
                    "freq": topk_counts,
                    "percent": sum(topk_counts) / total_counts,
                }

        return self.access_freq, self.weights_sum

    def dump_gate_score(self, path: str, layer=-1):
        # dump as json
        if layer != -1:
            data = {
                "topk": self.topk,
                "NK": [self.N, self.K],
                "num_layers": layer,
                "num_tokens": self.num_tokens,
                "num_samples": len(self.input_samples),
                "num_shared_experts": self.num_shared_experts,
                "access_freq": self.access_freq[layer].numpy().tolist(),
                "weights_sum": self.weights_sum[layer].numpy().tolist(),
                "percentile_stats": self.percentile_stats[layer],
            }
        else:
            data = {
                "topk": self.topk,
                "NK": [self.N, self.K],
                "num_layers": self.num_layers,
                "num_tokens": self.num_tokens,
                "num_samples": len(self.input_samples),
                "num_shared_experts": self.num_shared_experts,
            }
            data.update(
                {
                    f"layer-{i}": {
                        "access_freq": self.access_freq[i].numpy().tolist(),
                        "weights_sum": self.weights_sum[i].numpy().tolist(),
                        "percentile_stats": self.percentile_stats[i],
                    }
                    for i in range(0, self.num_layers)
                }
            )

        with open(path, "w") as f:
            options = jsbeautifier.default_options()
            options.indent_size = 2
            f.write(jsbeautifier.beautify(json.dumps(data), options))

    def reset_gate_score(self):
        # output of the MoE Gate of each layer: [layer, n_sample * topk_idx_weight]
        self.topk_idx = []
        self.topk_weight = []
        self.gate_hook_handle: list[torch.utils.hooks.RemovableHandle] = []

        # accumulate the topk_idx and topk_weight: [layer, Tensor[num_experts]]
        self.access_freq: list[Tensor] = []
        self.weights_sum: list[Tensor] = []

        if self.model_config.model_type == "deepseek_v2":
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 4, 8, 16, 32, 48]} for _ in range(self.num_layers)]
        elif self.model_config.model_type == "mixtral":
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 2, 4, 6]} for _ in range(self.num_layers)]
        elif self.model_config.model_type == "qwen2_moe":
            self.percentile_stats: list[dict] = [{i: {} for i in [1, 4, 8, 15, 30, 45]} for _ in range(self.num_layers)]

    def set_input(self, input_samples: list[Tensor]):
        # [1, seqlen]
        self.input_samples = input_samples
        self.num_tokens = input_samples[0].shape[1]

    @torch.no_grad()
    def collect_gate_score(self, path: str, layer_idx=-1):
        assert len(self.input_samples) > 0, "data not available."

        self.reset_gate_score()
        self.plug_gate_hook()
        self.trace_gate()
        self.clear_gate_hook()
        self.dump_gate_score(path, layer_idx)

    def gen_groupgemm_workload(self, save_path: str, layer_idx: int):
        experts_freq = self.access_freq[layer_idx]
        with open(save_path, "w") as f:
            for freq in experts_freq:
                if freq == 0:
                    continue
                f.write(f"16-16 {freq}x{self.N*2}x{self.K}\n")

    def plot_gate_score_layer(self, save_path: str, layer_idx: int):
        access_freq = self.access_freq[layer_idx]
        weights_sum = self.weights_sum[layer_idx]

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))

        axs[0].bar(range(self.num_experts), access_freq.numpy(), color="skyblue")
        axs[0].set_title(f"Access Frequency per Expert (Total {self.num_tokens} tokens)")
        axs[0].set_xlabel("Expert ID")
        axs[0].set_ylabel("Access Frequency")
        axs[0].set_xticks(range(self.num_experts))
        axs[0].set_xticklabels(range(self.num_experts), rotation=45)

        axs[1].bar(range(self.num_experts), weights_sum.numpy(), color="salmon")
        axs[1].set_title(f"Weights Sum per Expert (Total {self.num_tokens} tokens)")
        axs[1].set_xlabel("Expert ID")
        axs[1].set_ylabel("Weights Sum")
        axs[1].set_xticks(range(self.num_experts))
        axs[1].set_xticklabels(range(self.num_experts), rotation=45)

        fig.tight_layout()
        fig.savefig(save_path)

    def plot_gate_score_all(self, save_path: str = None):
        access_freq = self.access_freq
        weights_sum = self.weights_sum

        # plot two heatmap(freq and weight): [num_layers, num_experts]
        fig, axs = plt.subplots(2, 1, figsize=(self.num_layers * 1.2, self.num_layers))

        # plot access frequency heatmap
        freq_matrix = torch.stack(access_freq)  # shape: [num_layers, num_experts]
        sns.heatmap(
            freq_matrix.numpy(),
            ax=axs[0],
            cmap="Blues",
            cbar=True,
            xticklabels=range(self.num_experts),
            yticklabels=range(self.num_layers),
            annot=True,
            fmt="d",
        )
        axs[0].set_title(f"Access Frequency per Expert (Total {self.num_tokens} tokens)")
        axs[0].set_xlabel("Expert ID")
        axs[0].set_ylabel("Layer Index")

        # plot weights sum heatmap
        weight_matrix = torch.stack(weights_sum)  # shape: [num_layers, num_experts]
        sns.heatmap(
            weight_matrix.numpy(),
            ax=axs[1],
            cmap="Reds",
            cbar=True,
            xticklabels=range(self.num_experts),
            yticklabels=range(self.num_layers),
            annot=True,
            fmt=".2f",
        )
        axs[1].set_title(f"Weights Sum per Expert (Total {self.num_tokens} tokens)")
        axs[1].set_xlabel("Expert ID")
        axs[1].set_ylabel("Layer Index")

        plt.tight_layout()
        fig.savefig(save_path)

    def get_mlp_weights(self, layer_indices: int | list[int] = 0):
        if isinstance(layer_indices, int) and layer_indices == -1:
            layer_indices = range(self.num_layers)
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        model_name = self.model_config.model_type

        weights: dict[int, list[list[np.ndarray]]] = {}

        for layer_idx, layer_experts in enumerate(self.experts):
            if layer_idx not in layer_indices:
                continue
            layer_weights: list[list[np.ndarray]] = []

            for expert in layer_experts:
                expert_weight = []
                for _, w in expert.named_parameters():
                    expert_weight.append(w.data.float().cpu().numpy())
                layer_weights.append(expert_weight)

            weights[layer_idx] = layer_weights

        return weights

    @torch.no_grad()
    def plot_tensor_distribution(self, save_prefix: str, inp_tensors: dict[int, list[list[np.ndarray]]]):
        """
        input tensors: [layer_id, expert_id, tensor_id, tensor]
        """
        print(f"Plotting tensors distribution ...")

        for layer_idx, layer_tensors in tqdm(inp_tensors.items(), desc="Plotting layer"):
            num_experts = len(layer_tensors)

            for expert_id in tqdm(range(num_experts), leave=False, desc=f"Plotting experts in Layer-{layer_idx}"):
                # skip if expert is not activated
                if len(layer_tensors[expert_id]) == 0:
                    continue

                fig = plt.figure(figsize=(12, 4))
                fig_name = f"{save_prefix}L-{layer_idx}-E-{expert_id}.png"
                if not os.path.exists(os.path.dirname(fig_name)):
                    os.makedirs(os.path.dirname(fig_name))
                for i, wname in enumerate(["gate", "up", "down"]):
                    tensor = layer_tensors[expert_id][i]
                    sub_title = f"layer-{layer_idx} E-{expert_id}: {wname}-{list(tensor.shape)}"

                    M, N = tensor.shape
                    x = np.linspace(0, M - 1, M)
                    y = np.linspace(0, N - 1, N)
                    X, Y = np.meshgrid(x, y)

                    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
                    # surf = ax.plot_surface(X, Y, tensor.T, cmap='hot')
                    surf = ax.plot_surface(X, Y, tensor.T, cmap="coolwarm")

                    ax.xaxis.set_tick_params(pad=-5)
                    ax.yaxis.set_tick_params(pad=-3)
                    # ax.zaxis.set_tick_params(pad=-130)

                    # Adding labels
                    ax.set_xlabel("Output Channel")
                    ax.set_ylabel("Input Channel")
                    ax.set_zlabel("Value")
                    ax.set_title(sub_title)

                fig.tight_layout()
                fig.subplots_adjust(wspace=0.1)
                fig.savefig(fig_name, dpi=400)

    @torch.no_grad()
    def trace_activation(self, inp_samples: list[Tensor] | Tensor, layer_indices: int | list[int]):
        if isinstance(inp_samples, Tensor):
            inp_samples: list[Tensor] = [inp_samples]
        if isinstance(layer_indices, int) and layer_indices == -1:
            layer_indices = range(self.num_layers)
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        # [layer_id, expert_id, weight_id, tensor]
        self.act_stat: dict[int, list[list[np.ndarray]]] = {i: [] for i in range(self.num_layers)}

        act_stat_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        def stat_inp_act(m, inp, store: list):
            hidden_size = inp[0].shape[-1]
            store.append(inp[0].abs().reshape(-1, hidden_size).float().cpu().numpy().T)

        model_experts = get_expert_linears(self.model, layer_idx=-1, exclude_non_moe_layer=False)

        for layer_idx, layer_experts in enumerate(model_experts):
            if layer_idx not in layer_indices:
                continue

            self.act_stat[layer_idx] = [[] for _ in range(len(layer_experts))]
            for exp_idx, expert in enumerate(layer_experts):
                self.act_stat[layer_idx][exp_idx] = [[] for _ in range(len(["gate", "up", "down"]))]
                w_cnt = 0
                for _, m in expert.named_modules():
                    if not isinstance(m, nn.Linear):
                        continue
                    act_stat_hook = partial(stat_inp_act, store=self.act_stat[layer_idx][exp_idx][w_cnt])
                    act_stat_hook_handles.append(m.register_forward_pre_hook(act_stat_hook))
                    w_cnt += 1

        with torch.no_grad():
            for inp in inp_samples:
                self.model.model(inp.to(self.model.device))

            for layer_idx, layer_stat in self.act_stat.items():
                for exp_idx, exp_stat in enumerate(layer_stat):
                    for i, inp_stat in enumerate(exp_stat):
                        self.act_stat[layer_idx][exp_idx][i] = np.concatenate(inp_stat, axis=1)

        for handle in act_stat_hook_handles:
            handle.remove()

        return self.act_stat


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 python mxmoe/calibration/moe_tracer.py --model qwen2_moe --trace_gate
    """

    from project_config import ID2NAME

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ds2", help="Model ID")
    parser.add_argument("--layer", type=int, default=10, help="Layer index.")
    parser.add_argument("--dataset", type=str, default="wiki2", choices=["wiki2", "humaneval-x"], help="Sample dataset")
    parser.add_argument("--seqlen", type=int, default=4096, help="Input length")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None,
        help="GPU budget for accelerate layer placement (e.g. 20GiB).",
    )
    parser.add_argument(
        "--max-cpu-memory", type=str, default=None,
        help="CPU budget for accelerate layer placement (e.g. 64GiB).",
    )
    parser.add_argument(
        "--offload-folder", type=str, default=None,
        help="Disk offload folder for accelerate when GPU+CPU are insufficient.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trace_input", action="store_true", help="Trace activation")
    group.add_argument("--trace_gate", action="store_true", help="Trace MoE Gate(get freq and weight)")

    args = parser.parse_args()
    dataset = args.dataset
    model_id = args.model
    seqlen = args.seqlen
    nsamples = args.nsamples
    seed = args.seed
    target_layer = args.layer

    ############################################################################
    figure_path = f"{CUR_DIR}/figure"
    tokenizer = load_tokenizer(model_id)
    model = load_hf_model(
        model_id,
        max_memory=build_max_memory(args.max_gpu_memory, args.max_cpu_memory),
        offload_folder=args.offload_folder,
    )
    ############################################################################
    tracer = MoETracer(model)
    if dataset == "wiki2":
        trainloader, _ = get_wikitext2(nsamples, seed, seqlen, tokenizer, model_id)
    elif dataset == "humaneval-x":
        trainloader, _ = get_humaneval_x(nsamples, seed, seqlen, tokenizer, model_id)
    else:
        raise ValueError(f"Unsupported dataset `{dataset}`.")

    if args.trace_input:
        # tracer.plot_tensor_distribution(f"{figure_path}/weights/{model_id}", tracer.get_mlp_weights(1))
        tracer.trace_activation(trainloader, target_layer)

        # plot input activation distribution
        tracer.plot_tensor_distribution(f"{figure_path}/{model_id}/act/", tracer.act_stat)

    if args.trace_gate:
        print("Tracing MoE Gate ...")
        # for inp_len in tqdm([16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
        for inp_len in tqdm([4096]):
            inps: list[Tensor] = [x[:, :inp_len] for x in trainloader[:nsamples]]
            # print(inp)
            out_dir = f"{CUR_DIR}/calib/gate/{model_id}/{dataset}/{inp_len}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plot_layer = target_layer
            tracer.set_input(inps)
            tracer.collect_gate_score(f"{out_dir}/moe-gate.json")
            tracer.plot_gate_score_layer(f"{out_dir}/layer-{plot_layer}", plot_layer)
            tracer.plot_gate_score_all(f"{out_dir}/heatmap")
            # tracer.gen_groupgemm_workload(f"{out_dir}/{model_id}-workload-{inp_len}.txt", plot_layer)

        print("Tracing MoE Gate Done.")
