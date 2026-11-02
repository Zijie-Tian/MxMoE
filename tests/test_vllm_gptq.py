"""GPTQ export, numerical quality, and real Humming/MxMoE regressions."""

import inspect
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from mxmoe.quant.export_gptq_hf import (
    _exported_config_dict,
    apply_dense_ffn_group_padding,
    build_compressed_tensors_state_dict,
    require_compressed_tensors_alignment,
)
from mxmoe.quant.gptq_pack import (
    build_compressed_tensors_config,
    dense_ffn_alignment,
    pad_quant_state_for_group_alignment,
    quant_state_to_compressed_tensors,
)

from mxmoe.vllm_gptq_audit import _role, expected_humming_counts, mlp_layout
from run_vllm_gptq import (
    QualityDataError,
    aggregate_window_nlls,
    engine_lengths,
    length_failure,
    load_quality_data,
    resolve_shape_expectations,
    score_window_logprobs,
)

BITS, GROUP, HIDDEN, INTER, PAD = 2, 16, 16, 24, 8
DENSE = tuple(f"model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down"))
EXPERT = "model.layers.1.mlp.experts.0.gate_proj"
SHARED = "model.layers.0.mlp.shared_experts.gate_proj"


def _decode(words, width):
    # Scalar oracle, independent of the production unpacker. Unpack last axis.
    return torch.tensor([
        [((int(row[i // 16]) & 0xFFFFFFFF) >> (2 * (i % 16))) & 3 for i in range(width)]
        for row in words.tolist()
    ], dtype=torch.int32)


def _state(n, k, salt=1):
    row, col = torch.arange(n)[:, None], torch.arange(k)[None, :]
    groups = (k + GROUP - 1) // GROUP
    q = ((3 * row + 5 * col + salt) % 4).int()
    zero = ((row + 2 * torch.arange(groups)[None, :] + salt) % 4).int()
    # Deliberately not representable as fp16.
    scale = 1 + (row.float() + 1) * 2**-11 + torch.arange(groups)[None, :] * 2**-12
    g_idx = (torch.arange(k) // GROUP).int()
    weight = scale[:, g_idx.long()] * (q - zero[:, g_idx.long()])
    return q, zero, {"weight": weight, "scale": scale, "zero": zero.float(), "g_idx": g_idx}


def _pack(state):
    return quant_state_to_compressed_tensors(state, BITS, symmetric=False, groupsize=GROUP)


def _dequant_packed(packed):
    n, k = packed["weight_shape"].tolist()
    q = _decode(packed["weight_packed"], k)
    zero = _decode(packed["weight_zero_point"].T, n).T
    groups = torch.arange(k) // GROUP
    return packed["weight_scale"][:, groups] * (q - zero[:, groups])


def _pad(state, kind, pad=PAD):
    return pad_quant_state_for_group_alignment(state, kind=kind, pad=pad, bits=BITS, groupsize=GROUP)


def _linear(n, k):
    module = nn.Module()
    module.weight = nn.Parameter(torch.arange(n * k, dtype=torch.float16).reshape(n, k) / 128)
    return module


def _model_and_states():
    model = nn.Module()
    model.config = PretrainedConfig(intermediate_size=INTER, moe_intermediate_size=1408, first_k_dense_replace=1)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    dense, moe = model.model.layers
    dense.mlp, moe.mlp = nn.Module(), nn.Module()
    states = {}
    for name, shape, salt in zip(DENSE, [(INTER, HIDDEN), (INTER, HIDDEN), (HIDDEN, INTER)], (2, 3, 1)):
        setattr(dense.mlp, name.rsplit(".", 1)[-1], _linear(*shape))
        states[name] = _state(*shape, salt)[2]
    dense.mlp.shared_experts = nn.Module()
    dense.mlp.shared_experts.gate_proj = _linear(HIDDEN, HIDDEN)
    moe.mlp.experts = nn.ModuleList([nn.Module()])
    moe.mlp.experts[0].gate_proj = _linear(HIDDEN, HIDDEN)
    for name in (EXPERT, SHARED):
        states[name] = _state(HIDDEN, HIDDEN)[2]
    dense.self_attn = nn.Module()
    dense.self_attn.q_proj = _linear(HIDDEN, HIDDEN)
    return model, states


def test_unsigned_codes_raw_zeros_and_fp32_scales_survive_packing():
    codes, zeros, state = _state(32, 48)
    packed = _pack(state)
    assert packed["weight_packed"].dtype == torch.int32
    assert packed["weight_packed"].shape == (32, 3)
    assert packed["weight_zero_point"].shape == (2, 3)
    assert packed["weight_shape"].tolist() == [32, 48]
    assert torch.equal(_decode(packed["weight_packed"], 48), codes)
    # Raw zeros, not GPTQ-v1 zero-1 or a signed offset.
    assert torch.equal(_decode(packed["weight_zero_point"].T, 32).T, zeros)
    assert packed["weight_scale"].dtype == torch.float32
    assert torch.equal(packed["weight_scale"].view(torch.int32), state["scale"].view(torch.int32))
    torch.testing.assert_close(_dequant_packed(packed), state["weight"], rtol=0, atol=0)


@pytest.mark.parametrize("kind,n,k", [("output_rows", 24, 32), ("input_cols", 32, 24)])
def test_padding_preserves_codes_and_encodes_zero_as_q_equals_z(kind, n, k):
    codes, zeros, state = _state(n, k, salt=2)
    before = deepcopy(state)
    padded = _pad(state, kind)
    packed = _pack(padded)
    padded_n, padded_k = packed["weight_shape"].tolist()
    q = _decode(packed["weight_packed"], padded_k)
    z = _decode(packed["weight_zero_point"].T, padded_n).T
    assert torch.equal(q[:n, :k], codes)
    assert torch.equal(z[:n], zeros)
    for key in state:
        assert torch.equal(state[key], before[key])
    expected_zero_codes = z[:, torch.arange(padded_k) // GROUP]
    added = (slice(n, None), slice(None)) if kind == "output_rows" else (slice(None), slice(k, None))
    assert torch.equal(q[added], expected_zero_codes[added])
    assert torch.count_nonzero(_dequant_packed(packed)[added]) == 0


def test_separate_gate_up_padding_preserves_ffn_output():
    parts = [_state(INTER, HIDDEN, 2)[2], _state(INTER, HIDDEN, 3)[2], _state(HIDDEN, INTER)[2]]
    x = torch.arange(2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN) / 8

    def forward(weights):
        gate, up, down = weights
        return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T

    reference = forward([s["weight"] for s in parts])
    weights = [_dequant_packed(_pack(_pad(s, kind))) for s, kind in
               zip(parts, ("output_rows", "output_rows", "input_cols"))]
    torch.testing.assert_close(forward(weights), reference)
    # Padding after concatenation shifts up rows into gate; it must not be equivalent.
    merged = torch.cat([parts[0]["weight"], parts[1]["weight"], torch.zeros(2 * PAD, HIDDEN)])
    bad = forward([merged[:INTER + PAD], merged[INTER + PAD:], weights[2]])
    assert not torch.allclose(bad, reference)


def test_export_padding_keeps_source_and_experts_unchanged():
    model, states = _model_and_states()
    original = deepcopy(states)
    weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    report = apply_dense_ffn_group_padding(model, states, bits=BITS, groupsize=GROUP)
    assert {item["name"] for item in report["modules"]} == set(DENSE)
    assert report["exported_intermediate_size"] == INTER + PAD
    assert model.config.intermediate_size == INTER
    assert model.config.moe_intermediate_size == 1408
    for name in (EXPERT, SHARED):
        for key in states[name]:
            assert torch.equal(states[name][key], original[name][key])
    for name, param in model.named_parameters():
        assert torch.equal(param, weights[name])

    config = _exported_config_dict(model, intermediate_size=INTER + PAD,
                                  quantization_config=build_compressed_tensors_config(bits=BITS, group_size=GROUP, sym=False))
    packed, _ = build_compressed_tensors_state_dict(model, states, BITS, symmetric=False, groupsize=GROUP)
    assert config["intermediate_size"] == INTER + PAD
    assert config["moe_intermediate_size"] == 1408
    expected_shapes = ([INTER + PAD, HIDDEN], [INTER + PAD, HIDDEN], [HIDDEN, INTER + PAD])
    for name, shape in zip(DENSE, expected_shapes):
        assert name + ".weight" not in packed
        assert packed[name + ".weight_shape"].tolist() == shape
    assert not any("gate_up" in name for name in packed)
    assert "model.layers.0.self_attn.q_proj.weight" in packed
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in packed


def test_padding_without_a_dense_layer_fails_without_mutation():
    model, states = _model_and_states()
    model.config.first_k_dense_replace = 0
    before = deepcopy(states)
    with pytest.raises(ValueError):
        apply_dense_ffn_group_padding(model, states, bits=BITS, groupsize=GROUP)
    for name in states:
        for key in states[name]:
            assert torch.equal(states[name][key], before[name][key])


def test_reordered_groups_are_rejected_for_packing_and_padding():
    state = _state(32, 32)[2]
    state["g_idx"] = state["g_idx"].flip(0)
    with pytest.raises(ValueError):
        _pack(state)
    partial = _state(16, 24)[2]
    partial["g_idx"][0], partial["g_idx"][20] = 1, 0
    with pytest.raises(ValueError):
        _pad(partial, "input_cols")


@pytest.mark.parametrize("kind,pad", [("input_cols", 4), ("both", PAD)])
def test_unfilled_or_unsupported_padding_is_rejected(kind, pad):
    with pytest.raises(ValueError):
        _pad(_state(16, 24)[2], kind, pad)


def test_unaligned_shapes_and_unsupported_bits_are_rejected():
    with pytest.raises(ValueError):
        require_compressed_tensors_alignment({DENSE[2]: {"weight": torch.zeros(HIDDEN, INTER)}},
                                            bits=BITS, groupsize=GROUP, symmetric=False)
    short = _state(24, 16)[2]
    with pytest.raises(ValueError):
        _pack(short)
    with pytest.raises(NotImplementedError):
        quant_state_to_compressed_tensors(short, 3, symmetric=False, groupsize=GROUP)
    assert dense_ffn_alignment(10944, bits=2, groupsize=128)["padded_intermediate_size"] == 11008


def test_ppl_excludes_first_token_and_weights_by_predicted_tokens():
    first = score_window_logprobs([10, 11, 12], [{10: -99}, {11: NS(logprob=-0.5)}, {12: -1.5}])
    second = score_window_logprobs([20, 21], [None, {21: 0.0}])
    result = aggregate_window_nlls([first, second])
    assert first["predicted_tokens"] == 2
    assert first["nll_sum"] == 2.0
    assert result["predicted_tokens"] == 3
    assert result["ppl"] == pytest.approx(math.exp(2.0 / 3))


@pytest.mark.parametrize("scores", [
    None, [None], [None, None], [None, {99: -0.1}],
    [None, {11.9: -0.1}], [None, {11: math.nan}],
    [None, {11: math.inf}], [None, {11: -math.inf}],
    [None, {11: -0.1}, {12: -0.2}],
])
def test_incomplete_or_nonfinite_scores_fail(scores):
    with pytest.raises(QualityDataError):
        score_window_logprobs([10, 11], scores)


@pytest.mark.parametrize("window", [
    {"predicted_tokens": 0, "nll_sum": 1.0},
    {"predicted_tokens": 2, "nll_sum": math.nan},
])
def test_invalid_ppl_aggregation_fails(window):
    with pytest.raises(QualityDataError):
        aggregate_window_nlls([window])


@pytest.mark.parametrize("field,value", [
    ("token_windows", [[1, 2, 3]]),
    ("token_windows", [[1, True, 3, 4]]),
    ("prompts", [""]),
    ("dataset", {"name": "wiki", "split": "test", "sequence_length": 4, "windows": 9}),
])
def test_quality_input_rejects_inconsistent_data(tmp_path, field, value):
    payload = {
        "dataset": {"name": "wiki", "split": "test", "sequence_length": 4},
        "prompts": ["A natural prompt"], "token_windows": [[1, 2, 3, 4]],
    }
    payload[field] = value
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(QualityDataError):
        load_quality_data(path)


@pytest.mark.parametrize("limit,prompt,refused", [(4096, 8, True), (4097, 8, False), (4097, 4040, True)])
def test_quality_context_includes_generation(limit, prompt, refused):
    failure = length_failure(limit, window_tokens=4096, prompt_tokens=prompt, output_tokens=64)
    assert (failure is not None) == refused


def test_context_budget_respects_explicit_limit():
    args = NS(prompt_len=512, batch_size=1, output_tokens=128,
              observation_output_tokens=4, smoke_output_tokens=8,
              quality_required_len=4097, max_model_len=0)
    assert engine_lengths(args) == (4097, 4097)
    args.max_model_len = 8192
    assert engine_lengths(args) == (8192, 8192)


@pytest.mark.parametrize("config,dense,moe,linear_count", [
    (NS(model_type="qwen2_moe", num_hidden_layers=24, num_experts=60,
        decoder_sparse_step=1, shared_expert_intermediate_size=5632), [], list(range(24)), 48),
    (NS(model_type="qwen2_moe", num_hidden_layers=4, num_experts=60,
        decoder_sparse_step=2, mlp_only_layers=[3], shared_expert_intermediate_size=5632), [0, 2, 3], [1], 8),
    (NS(model_type="deepseek_v2", num_hidden_layers=27, n_routed_experts=64,
        first_k_dense_replace=1, moe_layer_freq=1, n_shared_experts=2), [0], list(range(1, 27)), 54),
    (NS(model_type="deepseek_v2", num_hidden_layers=4, n_routed_experts=64,
        first_k_dense_replace=1, moe_layer_freq=2, n_shared_experts=2), [0, 1, 3], [2], 8),
])
def test_decoder_coverage_includes_shared_and_interleaved_dense(config, dense, moe, linear_count):
    layout = mlp_layout(config)
    assert layout["dense_layers"] == dense
    assert layout["moe_layers"] == moe
    assert expected_humming_counts(layout) == {
        "HummingLinearMethod": linear_count, "HummingMoEMethod": len(moe),
    }


@pytest.mark.parametrize("suffix,role", [
    ("shared_expert.gate_up_proj", "shared_expert"),
    ("shared_experts.down_proj", "shared_expert"),
    ("shared_expert_gate", "router"), ("gate", "router"),
    ("gate_up_proj", "dense_mlp"),
])
def test_audit_distinguishes_shared_mlp_and_unquantized_gates(suffix, role):
    assert _role(f"model.layers.0.mlp.{suffix}", torch.nn.Linear(4, 4)) == role


def test_shape_expectations_follow_checkpoint_and_explicit_override(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "intermediate_size": 5632, "moe_intermediate_size": 1408,
    }))
    assert resolve_shape_expectations(str(tmp_path), None, None) == (5632, 1408, [])
    assert resolve_shape_expectations(str(tmp_path), 0, 2816) == (0, 2816, [])


def test_missing_shape_metadata_does_not_disable_checks(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    intermediate, moe, failures = resolve_shape_expectations(str(tmp_path), None, None)
    assert intermediate is None and moe is None
    assert failures
    assert resolve_shape_expectations(str(tmp_path), 0, 0) == (0, 0, [])


def _humming_init(order):
    import importlib
    module = "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16"
    if order == "audit-first":
        importlib.import_module("mxmoe.vllm_gptq_audit")
    ct = importlib.import_module(module)

    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend

    config = FusedMoEConfig(
        num_experts=4, experts_per_token=2, hidden_dim=256, intermediate_size=256,
        num_local_experts=4, num_logical_experts=4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU, in_dtype=torch.bfloat16, device="cuda",
        routing_method=RoutingMethodType.Renormalize, moe_backend="humming",
    )
    weights = [QuantizationArgs(num_bits=bits, type="int", symmetric=False,
                               strategy="group", group_size=128) for bits in (4, 8)]
    if order == "wna16-first":
        for weight in weights:
            try:
                ct.CompressedTensorsWNA16MoEMethod(weight, None, config)
            except ValueError:
                pass
            else:
                raise AssertionError("Unpatched asymmetric W4/W8 unexpectedly accepted")
        importlib.import_module("mxmoe.vllm_gptq_audit")
    for weight in weights:
        method = ct.CompressedTensorsWNA16MoEMethod(weight, None, config)
        assert method.wna16_backend == WNA16MoEBackend.HUMMING
        assert method.group_size == 128
        print(order, weight.num_bits, method.wna16_backend.value, method.experts_cls.__name__)


@pytest.mark.parametrize("order", ["wna16-first", "audit-first"])
def test_humming_real_constructor_in_fresh_process(order):
    pytest.importorskip("humming")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("Humming initialization requires CUDA SM75+")
    code = inspect.getsource(_humming_init) + f"\n_humming_init({order!r})\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _mxmoe_case(bits):
    from mxmoe.vllm import ops
    from mxmoe.quant.gptq_pack import _pack_unsigned_last_dim

    generator = torch.Generator().manual_seed(821)
    experts, n, k, group = 3, 256, 256, 128
    codes = torch.randint(1 << bits, (experts, n, k), generator=generator, dtype=torch.int32)
    zeros = torch.randint(1 << bits, (experts, n, k // group), generator=generator, dtype=torch.int32)
    scales = torch.linspace(0.003, 0.021, experts * n * (k // group)).reshape(experts, n, -1) / (1 << bits)
    group_idx = torch.arange(k) // group
    codes[0] = zeros[0, :, group_idx]  # A whole expert is exactly zero.
    reference = ((codes - zeros[:, :, group_idx]).half() * scales.half()[:, :, group_idx]).float()
    qweight = _pack_unsigned_last_dim(codes, bits).cuda()
    qzeros = _pack_unsigned_last_dim(zeros.transpose(-1, -2).contiguous(), bits).transpose(-1, -2).contiguous().cuda()
    packed, scale_zero = ops.prepare_weights(qweight, scales.cuda(), qzeros, bits)
    inputs = torch.randn(17, k, generator=generator).half().cuda()
    offsets = torch.tensor([0, 5, 5, 17], dtype=torch.int64, device="cuda")
    output = torch.empty(17, n, dtype=torch.float16, device="cuda")
    scratch = ops.workspace(experts, inputs.device)
    return inputs, packed, scale_zero, offsets, output, scratch, reference


def _mxmoe_reference(inputs, reference, offsets):
    rows = inputs.cpu().float()
    expected = torch.empty(rows.shape[0], reference.shape[1])
    for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        expected[start:end] = rows[start:end] @ reference[expert].T
    return expected.half()


def _require_mxmoe_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("MxMoE native regression requires CUDA SM80+")


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_mxmoe_packed_gemm_preserves_raw_zeros_and_empty_experts(bits):
    _require_mxmoe_gpu()
    from mxmoe.vllm import ops

    inputs, packed, scales, offsets, output, scratch, reference = _mxmoe_case(bits)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.grouped_gemm_out(inputs, packed, scales, offsets, bits, output, scratch)
    torch.cuda.current_stream().wait_stream(stream)
    expected = _mxmoe_reference(inputs, reference, [0, 5, 5, 17])
    torch.testing.assert_close(output.cpu(), expected, rtol=2e-3, atol=2e-4)
    assert torch.count_nonzero(output[:5]).item() == 0


def test_mxmoe_graph_replay_uses_current_expert_offsets():
    _require_mxmoe_gpu()
    from mxmoe.vllm import ops

    inputs, packed, scales, offsets, output, scratch, reference = _mxmoe_case(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.grouped_gemm_out(inputs, packed, scales, offsets, 2, output, scratch)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ops.grouped_gemm_out(inputs, packed, scales, offsets, 2, output, scratch)
    for boundaries in ([0, 0, 7, 17], [0, 17, 17, 17]):
        offsets.copy_(torch.tensor(boundaries, dtype=torch.int64, device=offsets.device))
        output.fill_(float("nan"))
        graph.replay()
        expected = _mxmoe_reference(inputs, reference, boundaries)
        torch.testing.assert_close(output.cpu(), expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("flags", [
    ["--activation-bits", "4"],
    ["--activation-bits", "8"],
    ["--quantization", "humming"],
])
def test_mxmoe_runner_rejects_humming_only_precision_paths(tmp_path, monkeypatch, flags):
    from run_vllm_gptq import parse_args

    monkeypatch.setattr(sys, "argv", [
        "run_vllm_gptq.py", str(tmp_path), "--moe-backend", "mxmoe", *flags,
    ])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def _activation_qconfig(bits=4, shared=False):
    choices = ((4, 8, 16, 4), (8, 8, 4, 16), (4, 16, 8, 8))
    if bits == 8:
        choices = ((8, 8, 16, 8), (8, 16, 8, 16), (8, 16, 8, 8))
    if shared:
        choices = (choices[0], choices[0], choices[2])
    return {"1": {"experts": {
        str(e): {
            name: {"w_bits": bits, "w_gsize": 128, "w_sym": False,
                   "a_bits": choices[p][e], "a_gsize": -1, "a_sym": True,
                   "a_clip": [0.75, 1.0] if p == 0 or (shared and p == 1) else [1.0, 1.0]}
            for p, name in enumerate(("gate", "up", "down"))
        } for e in range(4)
    }}}


def test_mxmoe_activation_config_preserves_independent_projection_schemes(tmp_path):
    from mxmoe.vllm.activation import load_activation_plan

    path = tmp_path / "qconfig.json"
    source = _activation_qconfig()
    path.write_text(json.dumps(source))
    plan = load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, 4)
    assert tuple(s.bits for s in plan.gate.schemes) == (4, 8, 16, 4)
    assert tuple(s.bits for s in plan.up.schemes) == (8, 8, 4, 16)
    assert tuple(s.bits for s in plan.down.schemes) == (4, 16, 8, 8)
    assert not plan.gate_up_shared
    # Equal bits alone do not permit reuse when clipping differs.
    for expert in source["1"]["experts"].values():
        expert["up"]["a_bits"] = expert["gate"]["a_bits"]
    path.write_text(json.dumps(source))
    assert not load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, 4).gate_up_shared
    for expert in source["1"]["experts"].values():
        expert["up"] = deepcopy(expert["gate"])
    path.write_text(json.dumps(source))
    assert load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, 4).gate_up_shared


@pytest.mark.parametrize("field,value", [
    ("a_bits", None), ("a_bits", 2), ("a_sym", False), ("a_gsize", 128),
    ("a_clip", [0.0, 1.0]), ("a_clip", [float("nan"), 1.0]), ("w_bits", 8),
])
def test_mxmoe_activation_config_rejects_missing_or_incompatible_schemes(tmp_path, field, value):
    from mxmoe.vllm.activation import load_activation_plan

    source = _activation_qconfig()
    cfg = source["1"]["experts"]["0"]["gate"]
    if value is None:
        del cfg[field]
    else:
        cfg[field] = value
    path = tmp_path / "qconfig.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, 4)


def test_mxmoe_activation_config_rejects_w8a4(tmp_path):
    from mxmoe.vllm.activation import load_activation_plan

    source = _activation_qconfig(8)
    source["1"]["experts"]["0"]["gate"]["a_bits"] = 4
    path = tmp_path / "qconfig.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="W8A4"):
        load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, 8)


def _decode_signed_words(words, bits):
    unsigned = words.to(torch.int32) & 0xFFFF
    shifts = list(range(16 - bits, -1, -bits))
    values = torch.stack([(unsigned >> shift) & ((1 << bits) - 1) for shift in shifts], dim=-1)
    values = values.reshape(*words.shape[:-1], -1)
    return torch.where(values >= (1 << (bits - 1)), values - (1 << bits), values)


def test_mxmoe_w8a8_matches_integer_group_zero_correction_oracle():
    _require_mxmoe_gpu()
    from mxmoe.quant.gptq_pack import _pack_unsigned_last_dim
    from mxmoe.vllm import ops

    generator = torch.Generator().manual_seed(991)
    experts, rows, n, k, group = 3, 17, 256, 256, 128
    codes = torch.randint(256, (experts, n, k), generator=generator, dtype=torch.int32)
    zeros = torch.randint(256, (experts, n, k // group), generator=generator, dtype=torch.int32)
    scales = torch.pow(2.0, -(9 + torch.arange(experts * n * 2) % 5)).reshape(experts, n, 2)
    qweight = _pack_unsigned_last_dim(codes, 8).cuda()
    qzeros = _pack_unsigned_last_dim(
        zeros.transpose(-1, -2).contiguous(), 8
    ).transpose(-1, -2).contiguous().cuda()
    modes = torch.full((1, experts), 8, dtype=torch.int32, device="cuda")
    packed_weight, scale_zero = ops.prepare_weights_mixed(qweight, scales.cuda(), qzeros, 8, modes)

    decoded_weight = _decode_signed_words(
        packed_weight[:, 0].view(experts, n, k // 2).cpu(), 8
    )
    torch.testing.assert_close(decoded_weight, codes - 128, rtol=0, atol=0)
    torch.testing.assert_close(
        scale_zero[:, 0, :, :, 0].cpu(), scales.half().permute(0, 2, 1),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        scale_zero[:, 0, :, :, 1].cpu(), zeros.permute(0, 2, 1).half(),
        rtol=0, atol=0,
    )

    values = (torch.randn(rows, k, generator=generator) / 2).half()
    values[0].zero_()
    inputs = values.cuda()
    offsets = torch.tensor([0, 5, 5, rows], dtype=torch.int64, device="cuda")
    packed_input = torch.empty(rows, k // 2, dtype=torch.int16, device="cuda")
    activation_scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    activation_sum = torch.empty(rows, k // group, dtype=torch.int32, device="cuda")
    clips = torch.ones(experts, 2, dtype=torch.float32, device="cuda")
    ops.quantize_activations_out(
        inputs, offsets, modes[0], clips, packed_input, activation_scale,
        activation_sum, inputs, False,
    )
    expected_scale = values.float().abs().amax(dim=1) / 127
    expected_scale = torch.where(expected_scale == 0, torch.ones_like(expected_scale), expected_scale)
    activation_codes = (values.float() / expected_scale[:, None]).round().clamp(-127, 127).to(torch.int32)
    torch.testing.assert_close(
        _decode_signed_words(packed_input.cpu(), 8), activation_codes, rtol=0, atol=0,
    )
    torch.testing.assert_close(activation_scale.cpu(), expected_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        activation_sum.cpu(), activation_codes.view(rows, -1, group).sum(dim=-1).to(torch.int32),
        rtol=0, atol=0,
    )

    output = torch.empty(rows, n, dtype=torch.float16, device="cuda")
    ops.grouped_gemm_mixed_out(
        inputs, packed_weight, scale_zero, offsets, 8, modes[0], packed_input,
        activation_scale, activation_sum, output, ops.workspace(experts, "cuda"),
        0, n, 2,
    )
    expected = torch.empty(rows, n)
    group_index = torch.arange(k) // group
    centered = codes - zeros[:, :, group_index]
    for expert, (start, end) in enumerate(zip((0, 5, 5), (5, 5, rows))):
        accum = torch.zeros(end - start, n)
        for g in range(k // group):
            dot = (
                activation_codes[start:end, g * group:(g + 1) * group].to(torch.int64)[:, None, :]
                * centered[expert, :, g * group:(g + 1) * group].to(torch.int64)[None, :, :]
            ).sum(dim=-1).float()
            accum += dot * expected_scale[start:end, None] * scales[expert, :, g].half().float()[None, :]
        expected[start:end] = accum
    torch.testing.assert_close(output.cpu(), expected.half(), rtol=2e-3, atol=2e-4)


def test_mxmoe_activation_config_content_changes_kernel_hash(tmp_path):
    from vllm.config.kernel import KernelConfig

    path = tmp_path / "qconfig.json"
    path.write_text('{\"a_bits\":4}')
    config = KernelConfig(moe_backend="mxmoe", mxmoe_activation_config=str(path))
    first = config.compute_hash()
    path.write_text('{\"a_bits\":8}')
    assert config.compute_hash() != first


def _mixed_weight_case(bits, n, k, generator):
    from mxmoe.quant.gptq_pack import _pack_unsigned_last_dim

    codes = torch.randint(1 << bits, (4, n, k), generator=generator, dtype=torch.int32)
    zeros = torch.randint(1 << bits, (4, n, k // 128), generator=generator, dtype=torch.int32)
    # Exact power-of-two scales isolate integer/zero correction from A16 weight
    # rounding, and vary across both channels and weight groups.
    scales = torch.pow(2.0, -(10 + torch.arange(4 * n * (k // 128)) % 4)).reshape(4, n, -1)
    groups = torch.arange(k) // 128
    reference = (codes - zeros[:, :, groups]).float() * scales[:, :, groups]
    qw = _pack_unsigned_last_dim(codes, bits).cuda()
    qz = _pack_unsigned_last_dim(zeros.transpose(-1, -2).contiguous(), bits).transpose(-1, -2).contiguous().cuda()
    return qw, scales.cuda(), qz, reference


def _activation_reference(values, scheme):
    values = values.float()
    if scheme.bits == 16:
        return values
    lower = (-values.amin(dim=-1, keepdim=True)).clamp_min(0) * scheme.clip[0]
    upper = values.amax(dim=-1, keepdim=True).clamp_min(0) * scheme.clip[1]
    qmax = (1 << (scheme.bits - 1)) - 1
    scales = torch.maximum(lower, upper) / qmax
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return (values / scales).round().clamp(-qmax, qmax) * scales


@pytest.mark.parametrize("bits,shared", [(2, False), (4, False), (4, True), (8, False)])
def test_mxmoe_internal_mixed_activations_with_routing_graph_replay(tmp_path, monkeypatch, bits, shared):
    _require_mxmoe_gpu()
    from mxmoe.vllm import ops
    from mxmoe.vllm.activation import load_activation_plan
    from mxmoe.vllm.experts import MxMoEExperts, make_mxmoe_moe_quant_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import get_moe_permute_scratch
    from vllm.v1.worker import workspace

    monkeypatch.setattr(workspace, "_manager", None)
    workspace.init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    path = tmp_path / "qconfig.json"
    path.write_text(json.dumps(_activation_qconfig(bits, shared)))
    plan = load_activation_plan(str(path), "model.layers.1.mlp.experts", 4, bits)
    assert plan.gate_up_shared == shared
    device_plan = plan.to_device("cuda")
    generator = torch.Generator().manual_seed(281)
    qw1, s1, z1, ref1 = _mixed_weight_case(bits, 512, 256, generator)
    qw2, s2, z2, ref2 = _mixed_weight_case(bits, 256, 256, generator)
    w1, sz1 = ops.prepare_weights_mixed(qw1, s1, z1, bits, device_plan.bits[:2])
    w2, sz2 = ops.prepare_weights_mixed(qw2, s2, z2, bits, device_plan.bits[2:3])
    config = FusedMoEConfig(
        num_experts=4, experts_per_token=2, hidden_dim=256, intermediate_size=256,
        num_local_experts=4, num_logical_experts=4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU, in_dtype=torch.float16, device="cuda",
        routing_method=RoutingMethodType.Renormalize, moe_backend="mxmoe",
    )
    method = MxMoEExperts(config, make_mxmoe_moe_quant_config(sz1, sz2, bits, 128, device_plan))
    x_cpu = (torch.randn(65, 256, generator=generator) / 2).half()
    x_cpu[:2].zero_()
    x = x_cpu.cuda()
    ids_cpu = torch.tensor([[0, 2], [2, 3], [3, 0]])[torch.arange(65) % 3]
    ids = ids_cpu.cuda()
    routing = torch.tensor([0.375, 0.625], device="cuda").expand(65, 2).contiguous()
    shapes = method.workspace_shapes(65, 512, 256, 2, 4, 4, None, MoEActivation.SILU)
    ws13, ws2, output = [torch.empty(shape, device="cuda", dtype=torch.float16) for shape in shapes]
    # Prepare the execution-slot scratch outside graph capture.
    get_moe_permute_scratch(num_experts=4, num_local_experts=4, device=torch.device("cuda"))

    def run():
        method.apply(output, x, w1, w2, routing, ids, MoEActivation.SILU,
                     4, None, None, None, ws13, ws2, None, False)

    def reference(selected):
        result = torch.zeros(65, 256)
        for expert in range(4):
            rows, slots = (selected == expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            gate = (_activation_reference(x_cpu[rows], plan.gate.schemes[expert]) @ ref1[expert, :256].T).half()
            up = (_activation_reference(x_cpu[rows], plan.up.schemes[expert]) @ ref1[expert, 256:].T).half()
            mid = (torch.nn.functional.silu(gate.float()).half().float() * up.float()).half()
            down = (_activation_reference(mid, plan.down.schemes[expert]) @ ref2[expert].T).half()
            result.index_add_(0, rows, down.float() * torch.tensor([0.375, 0.625])[slots, None])
        return result.half()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    rtol, atol = (5e-3, 2e-3) if bits == 8 else (3e-3, 3e-5)
    torch.testing.assert_close(output.cpu(), reference(ids_cpu), rtol=rtol, atol=atol)
    assert torch.count_nonzero(output[:2]).item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    # Change previously empty experts and force 65-row/tail cases on replay.
    for selected in (torch.tensor([1, 2]).expand(65, 2), torch.tensor([0, 3]).expand(65, 2)):
        ids.copy_(selected)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output.cpu(), reference(selected), rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend,quantization,pipeline_size,policy", [
    ("MXMOE", "humming", 1, None),
    ("mxmoe", "compressed-tensors", 2, None),
    ("humming", "compressed-tensors", 1, "activation.json"),
])
def test_mxmoe_kernel_config_rejects_unsafe_deployment(backend, quantization, pipeline_size, policy):
    from vllm.config.kernel import KernelConfig

    config = KernelConfig(moe_backend="auto", mxmoe_activation_config=policy)
    # EngineArgs can override fields after dataclass validation.
    config.moe_backend = backend
    runtime = NS(
        model_config=NS(quantization=quantization, architectures=["DeepseekV2ForCausalLM"]),
        parallel_config=NS(pipeline_parallel_size=pipeline_size),
    )
    with pytest.raises(ValueError):
        config.set_platform_defaults(runtime)
