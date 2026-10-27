#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

#include "act_kernel.cuh"
#include "cta_gemm.cuh"
#include "helper.h"
#include "hz_fused.cuh"
#include "layout.cuh"
#include "ref_impl.cuh"

using namespace mxmoe_ref;
using torch::Tensor;

namespace mxmoe {
auto ck0 = group_gemm_cutlass_f16<tiled_gemm::RCR_FP16FP16FP16, tiled_gemm::TileConfig<128, 128, 64, 2, 4, 1, 3>>;
auto ck1 = group_gemm_cutlass_f16<tiled_gemm::RCR_FP16FP16FP16, tiled_gemm::TileConfig<128, 128, 32, 2, 2, 1, 4>>;
auto ck2 = group_gemm_cutlass_f16<tiled_gemm::RCR_FP16FP16FP16, tiled_gemm::TileConfig<128, 128, 64, 1, 4, 1, 3>>;

auto mxmoe_w4a16_w8a8_bs64   = hz_fused_w4a16_w8a8_bs64;
auto mxmoe_w4a16_w8a8_bs128  = hz_fused_w4a16_w8a8_bs128;
auto mxmoe_w4a16_w8a8_bs256  = hz_fused_w4a16_w8a8_bs256;
auto mxmoe_w4a16_w8a8_bs512  = hz_fused_w4a16_w8a8_bs512;
auto mxmoe_w4a16_w8a8_bs1024 = hz_fused_w4a16_w8a8_bs1024;
auto mxmoe_w4a16_w8a8_bs2048 = hz_fused_w4a16_w8a8_bs2048;
auto mxmoe_w4a16_w8a8_bs4096 = hz_fused_w4a16_w8a8_bs4096;

auto mxmoe_w4a4_w8a8_bs1024 = nullptr;
auto mxmoe_w4a4_w8a8_bs2048 = nullptr;
auto mxmoe_w4a4_w8a8_bs4096 = nullptr;
auto mxmoe_w4a4_w8a8_bs8192 = nullptr;

// auto silu_mul_quant = silu_and_mul_quant_kernel<8>;

};  // namespace mxmoe

auto gg_permute_inp(const torch::Tensor& hidden, const torch::Tensor& topk_ids, int E) {
  const int topk = topk_ids.size(1);

  // values_sorted: [num_tokens * topk,]
  auto [values_sorted, indices] = torch::sort(topk_ids.view({-1}));

  // [num_tokens,]: i-th value V: sorted_token[i] is from V-th token of hidden
  indices = indices.floor_divide(topk);

  // shape: [num_tokens * topk, K]
  torch::Tensor inp_buffer = hidden.index_select(0, indices);
  // [E,]
  torch::Tensor recv_tokens_per_exp = torch::bincount(values_sorted, {}, E).cpu();
  torch::Tensor nonzero_mask        = recv_tokens_per_exp != 0;
  int num_problems                  = nonzero_mask.sum().item<int64_t>();

  return std::make_tuple(num_problems, inp_buffer, indices, recv_tokens_per_exp);
}

void gg_unpermute_out() {}

torch::Tensor moe_cutlass(          //
    const torch::Tensor& inp,       // [bs, seq_len, K] or [num_tokens, K]
    const torch::Tensor& experts,   // [E, N, K]
    const torch::Tensor& topk_ids,  // [bs, seq_len, topk] or [num_tokens, topk]
    torch::Tensor& output,          // [bs * seq_len * topk, N] or [num_tokens * topk, N]

    const std::optional<int>& num_problems_                  = std::nullopt,  //
    const std::optional<torch::Tensor>& inp_buffer_          = std::nullopt,  //
    const std::optional<torch::Tensor>& perm_indices_        = std::nullopt,  //
    const std::optional<torch::Tensor>& recv_tokens_per_exp_ = std::nullopt   //
) {
  TORCH_CHECK(inp.dim() == 2, "hidden must be 2D");
  TORCH_CHECK(inp.device().is_cuda() && experts.device().is_cuda(), "dev must be on cuda");

  const int E    = experts.size(0);
  const int N    = experts.size(1);
  const int K    = experts.size(2);
  const int topk = topk_ids.size(1);

  const int num_tokens = inp.size(0);
  int num_problems     = 0;

  torch::Tensor recv_tokens_per_exp;
  torch::Tensor inp_buffer;
  torch::Tensor perm_indices;

  if (perm_indices_.has_value()) {
    num_problems        = num_problems_.value();
    inp_buffer          = inp_buffer_.value();
    perm_indices        = perm_indices_.value();
    recv_tokens_per_exp = recv_tokens_per_exp_.value();
  } else {
    auto perm_out       = gg_permute_inp(inp, topk_ids, E);
    num_problems        = std::get<0>(perm_out);
    inp_buffer          = std::get<1>(perm_out);
    perm_indices        = std::get<2>(perm_out);
    recv_tokens_per_exp = std::get<3>(perm_out);
  }

  // std::cout << "num_problems: " << num_problems << std::endl;
  // std::cout << "perm_indices" << perm_indices << std::endl;
  // std::cout << "recv_tokens_per_exp: " << recv_tokens_per_exp << std::endl;
  // std::cout << "inp_buffer: " << inp_buffer.sizes() << std::endl;
  // std::cout << "experts: " << experts.sizes() << std::endl;

  std::vector<at::Half*> ptr_inps(num_problems * 3);
  std::vector<int64_t> lds(num_problems * 3);
  std::vector<GemmCoord> h_problem_sizes(num_problems);

  auto ptr_As = ptr_inps.data() + num_problems * 0;
  auto ptr_Bs = ptr_inps.data() + num_problems * 1;
  auto ptr_Cs = ptr_inps.data() + num_problems * 2;
  auto ldas   = lds.data() + num_problems * 0;
  auto ldbs   = lds.data() + num_problems * 1;
  auto ldcs   = lds.data() + num_problems * 2;

  half** d_ptr_As            = nullptr;
  int64_t* d_ldas            = nullptr;
  GemmCoord* d_problem_sizes = nullptr;

  auto token_per_exp = recv_tokens_per_exp.data_ptr<int64_t>();
  int st = 0, problem_idx = 0;
  for (int exp_id = 0; exp_id < E; exp_id++) {
    int exp_recv_tokens = token_per_exp[exp_id];
    // std::cout << "exp_id: " << exp_id << " exp_recv_tokens: " << exp_recv_tokens << std::endl;
    if (exp_recv_tokens == 0) {
      continue;
    }

    // get start pointer of inp_buffer[st:st+i], append to ptr_As
    ptr_As[problem_idx] = inp_buffer.slice(0, st, st + exp_recv_tokens).data_ptr<at::Half>();
    // get start pointer of experts[exp_id], append to ptr_Bs
    ptr_Bs[problem_idx] = experts[exp_id].data_ptr<at::Half>();
    // get start pointer of out_buffer[st:st+i], append to ptr_Cs
    ptr_Cs[problem_idx]          = output.slice(0, st, st + exp_recv_tokens).data_ptr<at::Half>();
    ldas[problem_idx]            = K;
    ldbs[problem_idx]            = K;
    ldcs[problem_idx]            = N;
    h_problem_sizes[problem_idx] = GemmCoord(exp_recv_tokens, N, K);

    st += exp_recv_tokens;
    problem_idx++;
  }
  checkCudaErrors(cudaMalloc(&d_problem_sizes, sizeof(GemmCoord) * num_problems));
  checkCudaErrors(cudaMalloc(&d_ptr_As, sizeof(half*) * num_problems * 3));  // for abcd
  checkCudaErrors(cudaMalloc(&d_ldas, sizeof(int64_t) * num_problems * 3));  // for abcd

  checkCudaErrors(
      cudaMemcpy(d_problem_sizes, h_problem_sizes.data(), sizeof(GemmCoord) * num_problems, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ptr_As, ptr_As, sizeof(half*) * num_problems * 3, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ldas, ldas, sizeof(int64_t) * num_problems * 3, cudaMemcpyHostToDevice));

  auto d_ptr_Bs = d_ptr_As + num_problems;
  auto d_ptr_Cs = d_ptr_As + 2 * num_problems;
  auto d_ptr_Ds = d_ptr_As + 2 * num_problems;
  auto d_ldbs   = d_ldas + num_problems;
  auto d_ldcs   = d_ldas + 2 * num_problems;
  auto d_ldds   = d_ldas + 2 * num_problems;

  // std::cout << "num_problems: " << num_problems << std::endl;

  if (num_tokens <= 128) {
    mxmoe::ck1(d_ptr_As, d_ptr_Bs, d_ptr_Cs, d_ptr_Ds, d_ldas, d_ldbs, d_ldcs, d_ldds, d_problem_sizes,
               h_problem_sizes.data(), num_problems);
  } else {
    mxmoe::ck0(d_ptr_As, d_ptr_Bs, d_ptr_Cs, d_ptr_Ds, d_ldas, d_ldbs, d_ldcs, d_ldds, d_problem_sizes,
               h_problem_sizes.data(), num_problems);
  }

  checkCudaErrors(cudaFree(d_ptr_As));
  checkCudaErrors(cudaFree(d_ldas));
  checkCudaErrors(cudaFree(d_problem_sizes));

  return output;
}

torch::Tensor moe_cutlass_share_fused(    //
    const torch::Tensor& inp,             // [bs, seq_len, K]
    const torch::Tensor& experts,         // [E, N, K]
    const torch::Tensor& shared_experts,  // [N', K]
    const torch::Tensor& topk_ids,        // [bs, seq_len, topk] or [num_tokens, topk]
    torch::Tensor& output,                // [bs * seq_len * topk, N] or [num_tokens * topk, N]
    torch::Tensor& shared_output,         // [bs * seq_len * topk, N] or [num_tokens * topk, N]

    const std::optional<int>& num_problems_                  = std::nullopt,  //
    const std::optional<torch::Tensor>& inp_buffer_          = std::nullopt,  //
    const std::optional<torch::Tensor>& perm_indices_        = std::nullopt,  //
    const std::optional<torch::Tensor>& recv_tokens_per_exp_ = std::nullopt   //
) {
  TORCH_CHECK(inp.dim() == 2, "hidden must be 2D");
  TORCH_CHECK(inp.device().is_cuda() && experts.device().is_cuda(), "dev must be on cuda");

  const int E        = experts.size(0);
  const int N        = experts.size(1);
  const int K        = experts.size(2);
  const int N_shared = shared_experts.size(0);
  const int K_shared = shared_experts.size(1);
  const int topk     = topk_ids.size(1);

  const int num_tokens = inp.size(0);
  int num_problems     = 0;

  torch::Tensor recv_tokens_per_exp;
  torch::Tensor inp_buffer;
  torch::Tensor perm_indices;

  if (perm_indices_.has_value()) {
    num_problems        = num_problems_.value();
    inp_buffer          = inp_buffer_.value();
    perm_indices        = perm_indices_.value();
    recv_tokens_per_exp = recv_tokens_per_exp_.value();
  } else {
    auto perm_out       = gg_permute_inp(inp, topk_ids, E);
    num_problems        = std::get<0>(perm_out);
    inp_buffer          = std::get<1>(perm_out);
    perm_indices        = std::get<2>(perm_out);
    recv_tokens_per_exp = std::get<3>(perm_out);
  }

  // std::cout << "num_problems: " << num_problems << std::endl;
  // std::cout << "perm_indices" << perm_indices << std::endl;
  // std::cout << "recv_tokens_per_exp: " << recv_tokens_per_exp << std::endl;
  // std::cout << "inp_buffer: " << inp_buffer.sizes() << std::endl;
  // std::cout << "experts: " << experts.sizes() << std::endl;

  num_problems += 1;
  std::vector<at::Half*> ptr_inps(num_problems * 3);
  std::vector<int64_t> lds(num_problems * 3);
  std::vector<GemmCoord> h_problem_sizes(num_problems, {num_tokens, N_shared, K_shared});

  auto ptr_As = ptr_inps.data();
  auto ptr_Bs = ptr_inps.data() + num_problems * 1;
  auto ptr_Cs = ptr_inps.data() + num_problems * 2;
  auto ldas   = lds.data();
  auto ldbs   = lds.data() + num_problems * 1;
  auto ldcs   = lds.data() + num_problems * 2;

  half** d_ptr_As            = nullptr;
  int64_t* d_ldas            = nullptr;
  GemmCoord* d_problem_sizes = nullptr;

  auto token_per_exp = recv_tokens_per_exp.data_ptr<int64_t>();
  int st = 0, problem_idx = 0, exp_recv_tokens = 0;
  for (int exp_id = 0; exp_id < E; exp_id++) {
    exp_recv_tokens = token_per_exp[exp_id];
    // std::cout << "exp_id: " << exp_id << " exp_recv_tokens: " << exp_recv_tokens << std::endl;
    if (exp_recv_tokens == 0) {
      continue;
    }

    // get start pointer of inp_buffer[st:st+i], append to ptr_As
    ptr_As[problem_idx] = inp_buffer.slice(0, st, st + exp_recv_tokens).data_ptr<at::Half>();
    // get start pointer of experts[exp_id], append to ptr_Bs
    ptr_Bs[problem_idx] = experts[exp_id].data_ptr<at::Half>();
    // get start pointer of out_buffer[st:st+i], append to ptr_Cs
    ptr_Cs[problem_idx]          = output.slice(0, st, st + exp_recv_tokens).data_ptr<at::Half>();
    ldas[problem_idx]            = K;
    ldbs[problem_idx]            = K;
    ldcs[problem_idx]            = N;
    h_problem_sizes[problem_idx] = GemmCoord(exp_recv_tokens, N, K);

    st += exp_recv_tokens;
    problem_idx++;
  }
  ptr_As[problem_idx] = inp.data_ptr<at::Half>();
  ptr_Bs[problem_idx] = shared_experts.data_ptr<at::Half>();
  ptr_Cs[problem_idx] = shared_output.data_ptr<at::Half>();
  ldas[problem_idx]   = K;
  ldbs[problem_idx]   = K;
  ldcs[problem_idx]   = N_shared;

  checkCudaErrors(cudaMalloc(&d_problem_sizes, sizeof(GemmCoord) * num_problems));
  checkCudaErrors(cudaMalloc(&d_ptr_As, sizeof(half*) * num_problems * 3));  // for abcd
  checkCudaErrors(cudaMalloc(&d_ldas, sizeof(int64_t) * num_problems * 3));  // for abcd

  checkCudaErrors(
      cudaMemcpy(d_problem_sizes, h_problem_sizes.data(), sizeof(GemmCoord) * num_problems, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ptr_As, ptr_As, sizeof(half*) * num_problems * 3, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ldas, ldas, sizeof(int64_t) * num_problems * 3, cudaMemcpyHostToDevice));

  auto d_ptr_Bs = d_ptr_As + num_problems;
  auto d_ptr_Cs = d_ptr_As + 2 * num_problems;
  auto d_ptr_Ds = d_ptr_As + 2 * num_problems;
  auto d_ldbs   = d_ldas + num_problems;
  auto d_ldcs   = d_ldas + 2 * num_problems;
  auto d_ldds   = d_ldas + 2 * num_problems;

  // std::cout << "num_problems: " << num_problems << std::endl;

  if (num_tokens <= 128) {
    mxmoe::ck1(d_ptr_As, d_ptr_Bs, d_ptr_Cs, d_ptr_Ds, d_ldas, d_ldbs, d_ldcs, d_ldds, d_problem_sizes,
               h_problem_sizes.data(), num_problems);
  } else {
    mxmoe::ck0(d_ptr_As, d_ptr_Bs, d_ptr_Cs, d_ptr_Ds, d_ldas, d_ldbs, d_ldcs, d_ldds, d_problem_sizes,
               h_problem_sizes.data(), num_problems);
  }

  checkCudaErrors(cudaFree(d_ptr_As));
  checkCudaErrors(cudaFree(d_ldas));
  checkCudaErrors(cudaFree(d_problem_sizes));

  return output;
}

auto moe_mxmoe_share_fused(                  //
    const std::vector<Tensor>& inp,          // [bs, seq_len, K]
    const std::vector<Tensor>& experts,      // [E, N, K]
    const std::vector<Tensor>& scale_zp_a,   //
    const std::vector<Tensor>& scale_zp_b,   //
    std::vector<Tensor>& output,             //
    const int N, const int K,                //
    const int shared_N, const int shared_K,  //

    const std::vector<std::tuple<int, int, int, bool>>& qparams_per_exp,  //

    const torch::Tensor& recv_tokens_per_exp,  //
    bool verbose = false                       //
) {
  using mxmoe::QParams;

  const int E                       = recv_tokens_per_exp.size(0);
  const int num_tokens              = inp.back().size(0);
  const int num_problems            = inp.size();
  const bool has_shared_experts     = experts.size() > E;
  const int num_experts_incl_shared = E + (has_shared_experts ? 1 : 0);

  // std::cout << "num_problems: " << num_problems << std::endl;
  // std::cout << "recv_tokens_per_exp: " << recv_tokens_per_exp << std::endl;
  // std::cout << "inp_buffer: " << inp_buffer.sizes() << std::endl;

  // std::cout << "inp size: " << inp.size() << std::endl;
  // std::cout << "experts: " << experts.size() << std::endl;
  // std::cout << "scale_zp: " << scale_zp.size() << std::endl;

  std::vector<at::Half*> ptr_inps(num_problems * 5, nullptr);
  std::vector<dim3> h_problem_sizes(num_problems);
  std::vector<QParams> h_qparams_list(num_problems);

  auto ptr_As         = ptr_inps.data();
  auto ptr_Bs         = ptr_inps.data() + num_problems * 1;
  auto ptr_Cs         = ptr_inps.data() + num_problems * 2;
  auto ptr_scale_zp_a = ptr_inps.data() + num_problems * 3;
  auto ptr_scale_zp_b = ptr_inps.data() + num_problems * 4;

  half** d_ptr_As         = nullptr;
  dim3* d_problem_sizes   = nullptr;
  QParams* d_qparams_list = nullptr;

  auto token_per_exp = recv_tokens_per_exp.data_ptr<int64_t>();
  int st = 0, problem_idx = 0, exp_recv_tokens = 0;
  for (int e = 0; e < num_experts_incl_shared; e++) {
    exp_recv_tokens = (e == E) ? num_tokens : token_per_exp[e];
    if (exp_recv_tokens == 0) continue;

    auto cur_inp                      = inp[problem_idx];
    auto cur_weight                   = experts[e];
    auto cur_out                      = output[problem_idx];
    auto [abits, wbits, qgsize, qsym] = qparams_per_exp[e];

    ptr_As[problem_idx]          = reinterpret_cast<at::Half*>(cur_inp.data_ptr());
    ptr_Bs[problem_idx]          = reinterpret_cast<at::Half*>(cur_weight.data_ptr());
    ptr_Cs[problem_idx]          = reinterpret_cast<at::Half*>(cur_out.data_ptr());
    ptr_scale_zp_a[problem_idx]  = reinterpret_cast<at::Half*>(scale_zp_a[problem_idx].data_ptr());
    ptr_scale_zp_b[problem_idx]  = reinterpret_cast<at::Half*>(scale_zp_b[e].data_ptr());
    h_problem_sizes[problem_idx] = e == E ? dim3(exp_recv_tokens, shared_N, shared_K) : dim3(exp_recv_tokens, N, K);
    h_qparams_list[problem_idx]  = QParams(make_int2(abits, wbits), qgsize, qsym);

    // std::cout << "exp_id: " << exp_id << ", problem_idx: " << problem_idx << " exp_recv_tokens: " << exp_recv_tokens
    //           << " NK: " << N << "," << K << " [" << h_problem_sizes[problem_idx].x << " "
    //           << h_problem_sizes[problem_idx].y << " " << h_problem_sizes[problem_idx].z << "]" << std::endl;

    st += exp_recv_tokens;
    problem_idx++;
  }

  if (verbose) {
    std::cout << "num_problems: " << num_problems << std::endl;
    for (int i = 0; i < num_problems; i++) {
      std::cout << "problem " << i << ": [" << h_problem_sizes[i].x << "," << h_problem_sizes[i].y << ","
                << h_problem_sizes[i].z << "]" << std::endl;
      std::cout << "qparams: [" << h_qparams_list[i].qbits.x << " " << h_qparams_list[i].qbits.y << "] "
                << h_qparams_list[i].gsize << " " << h_qparams_list[i].sym << std::endl;
      std::cout << "input: [" << inp[i].size(0) << ", " << inp[i].size(1) << "]" << std::endl;
    }
  }

  // prepare kernel inputs
  checkCudaErrors(cudaMalloc(&d_ptr_As, sizeof(half*) * num_problems * 5));  // for abc, scale
  checkCudaErrors(cudaMalloc(&d_problem_sizes, sizeof(dim3) * num_problems));
  checkCudaErrors(cudaMalloc(&d_qparams_list, sizeof(QParams) * num_problems));

  checkCudaErrors(  //
      cudaMemcpy(d_ptr_As, ptr_As, sizeof(half*) * num_problems * 5, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_problem_sizes, h_problem_sizes.data(), sizeof(dim3) * num_problems, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_qparams_list, h_qparams_list.data(), sizeof(QParams) * num_problems, cudaMemcpyHostToDevice));

  auto d_ptr_Bs         = d_ptr_As + num_problems;
  auto d_ptr_Cs         = d_ptr_As + 2 * num_problems;
  auto d_ptr_Ds         = d_ptr_As + 2 * num_problems;
  auto d_ptr_scale_zp_a = d_ptr_As + 3 * num_problems;
  auto d_ptr_scale_zp_b = d_ptr_As + 4 * num_problems;

  mxmoe::mxmoe_w4a16_w8a8_bs256(  //
      d_ptr_As, d_ptr_Bs, d_ptr_scale_zp_a, d_ptr_scale_zp_b, d_ptr_Cs, d_ptr_Ds, nullptr, nullptr, nullptr, nullptr,
      d_problem_sizes, h_problem_sizes.data(), d_qparams_list, h_qparams_list.data(), num_problems);

  if (num_tokens <= 128) {
    // mxmoe::mxmoe_w4a16_w8a8_bs256(  //
    //     d_ptr_As, d_ptr_Bs, scale_zp.data_ptr<at::Half>(), d_ptr_Cs, d_ptr_Ds, nullptr, nullptr, nullptr, nullptr,
    //     d_problem_sizes, h_problem_sizes.data(), d_qparams_list, qparams_list.data(), num_problems);
  } else {
    // mxmoe::mxmoe_w4a16_w8a8_bs256(  //
    //     d_ptr_As, d_ptr_Bs, scale_zp.data_ptr<at::Half>(), d_ptr_Cs, d_ptr_Ds, nullptr, nullptr, nullptr, nullptr,
    //     d_problem_sizes, h_problem_sizes.data(), d_qparams_list, qparams_list.data(), num_problems);
  }

  checkCudaErrors(cudaFree(d_ptr_As));
  checkCudaErrors(cudaFree(d_problem_sizes));
  checkCudaErrors(cudaFree(d_qparams_list));

  return output;
}

/// @return q_inp_list, q_inp_scale_list, q_inp, q_inp_scale, recv_tokens_pre_exp, perm_indices
auto quant_inp_act(                //
    const Tensor& hidden,          // [bs, seq_len, K]
    const Tensor& topk_ids,        //
    const int num_experts,         //
    const int N,                   //
    const int num_shared_experts,  //

    const std::vector<std::tuple<int, int, int, bool>>& qparams_per_exp,  //
    const bool verbose = false) -> std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, Tensor,
                                              Tensor, Tensor, Tensor, Tensor, Tensor> {
  const int num_tokens          = hidden.size(0);
  const int K                   = hidden.size(1);
  const int64_t topk            = topk_ids.size(1);
  const bool has_shared_experts = num_shared_experts > 0;

  auto [dst_exp_sorted, perm_indices] = torch::sort(topk_ids.view({-1}));
  perm_indices                        = perm_indices.floor_divide(topk);
  Tensor recv_tokens_per_exp          = torch::bincount(dst_exp_sorted, {}, num_experts).cpu();
  Tensor nonzero_mask                 = recv_tokens_per_exp != 0;
  const int num_problems              = nonzero_mask.sum().item<int64_t>();

  auto options = torch::TensorOptions().dtype(torch::kF16).device(torch::kCUDA);

  int alloc_tokens              = num_tokens * topk + (has_shared_experts ? num_tokens : 0);
  auto num_exps_incl_shared     = num_experts + (has_shared_experts ? 1 : 0);
  auto num_problems_incl_shared = num_problems + (has_shared_experts ? 1 : 0);

  auto cvt_qparams_to_tag = [](int qbits, int gsize) {
    if (qbits >= 16) {
      return 0;
    } else if (qbits == 8 && gsize == -1) {
      return 1;
    } else if (qbits == 4 && gsize == -1) {
      return 2;
    } else if (qbits == 4 && gsize == 128) {
      return 3;
    } else {
      return -1;
    }
  };

  auto host_buffer         = std::vector<int>(num_exps_incl_shared * 4);
  auto exp_qtag            = host_buffer.data();
  auto exp_st_token_idx    = host_buffer.data() + num_exps_incl_shared;
  auto exp_qinp_write_off  = host_buffer.data() + 2 * num_exps_incl_shared;
  auto exp_scale_write_off = host_buffer.data() + 3 * num_exps_incl_shared;

  int acc_tokens           = 0;
  int qinp_write_off       = 0;
  int qinp_scale_write_off = 0;

  auto qinp_sizes       = std::vector<int>(num_exps_incl_shared);
  auto qinp_scale_sizes = std::vector<int>(num_exps_incl_shared);

  auto tokens_per_exp = recv_tokens_per_exp.data_ptr<int64_t>();
  for (int e = 0; e < num_exps_incl_shared; e++) {
    auto cur_tag            = cvt_qparams_to_tag(std::get<0>(qparams_per_exp[e]), std::get<2>(qparams_per_exp[e]));
    auto cur_qinp_write_ld  = cur_tag == 0 ? K : (cur_tag == 1 ? K / 2 : K / 4);
    auto cur_scale_write_ld = cur_tag == 0 ? 0 : ((cur_tag == 1 || cur_tag == 2) ? 1 : K / 128);
    auto cur_exp_tokens     = (e == num_experts) ? num_tokens : tokens_per_exp[e];

    exp_qtag[e]            = cur_tag;
    exp_st_token_idx[e]    = acc_tokens;
    exp_qinp_write_off[e]  = qinp_write_off;
    exp_scale_write_off[e] = qinp_scale_write_off;
    qinp_sizes[e]          = cur_qinp_write_ld * cur_exp_tokens;
    qinp_scale_sizes[e]    = cur_scale_write_ld * cur_exp_tokens;

    qinp_write_off += cur_qinp_write_ld * cur_exp_tokens;
    qinp_scale_write_off += cur_scale_write_ld * cur_exp_tokens;
    acc_tokens += cur_exp_tokens;
  }

  size_t tot_bytes     = sizeof(int) * num_exps_incl_shared * 4;
  int* d_combined_buff = nullptr;
  checkCudaErrors(cudaMalloc(&d_combined_buff, tot_bytes));
  int offset      = 0;
  int* d_exp_qtag = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_st_token_idx = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_qinp_write_off = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_scale_write_off = d_combined_buff + offset;
  checkCudaErrors(cudaMemcpy(d_combined_buff, host_buffer.data(), tot_bytes, cudaMemcpyHostToDevice));

  auto inp_store       = torch::empty({qinp_write_off}, options);
  auto inp_scale_store = torch::empty({qinp_scale_write_off}, options);
  auto out_store       = torch::empty({(num_shared_experts + topk) * num_tokens * N * 2}, options);

  // auto block = std::min(K / 2, 1024);
  auto block = 1024;
  // determine necessary storage size:
  auto block_reduce_temp_bytes = sizeof(typename cub::BlockReduce<half, 1024>::TempStorage);
  // finally, we need to make sure that we can hold at least one half
  // needed in the kernel to exchange data after reduction
  auto smem_size = (std::max)(1 * sizeof(half), block_reduce_temp_bytes);
  quant_act_kernel<<<alloc_tokens, block, smem_size>>>(  //
      (const half*)hidden.data_ptr<at::Half>(),          //
      (half*)inp_store.data_ptr<at::Half>(),             //
      (half*)inp_scale_store.data_ptr<at::Half>(),       //
      dst_exp_sorted.data_ptr<int>(),                    //
      perm_indices.data_ptr<int64_t>(),                  //
      d_exp_qtag,                                        //
      d_exp_st_token_idx,                                //
      d_exp_qinp_write_off,                              //
      d_exp_scale_write_off,                             //
      K,                                                 //
      num_tokens * topk,                                 //
      num_experts                                        //
  );

  auto inp_list       = std::vector<Tensor>{};
  auto inp_scale_list = std::vector<Tensor>{};
  auto out_list       = std::vector<Tensor>{};
  inp_list.reserve(num_problems_incl_shared);
  inp_scale_list.reserve(num_problems_incl_shared);
  out_list.reserve(num_problems_incl_shared);

  checkCudaErrors(cudaFree(d_combined_buff));

  if (verbose) {
    for (int e = 0; e < num_exps_incl_shared; e++) {
      auto pack_num   = exp_qtag[e] == 0 ? 1 : (exp_qtag[e] == 1 ? 2 : 4);
      auto exp_tokens = (e == num_experts) ? num_tokens : tokens_per_exp[e];
      if (exp_tokens == 0) continue;
      std::cout << "num_exps_incl_shared: " << num_exps_incl_shared << " exp: " << e << " qinp_size: " << qinp_sizes[e]
                << " qinp_scale_size: " << qinp_scale_sizes[e] << " qinp: [" << exp_tokens << ", " << K / pack_num
                << "] " << std::endl;
    }
  }
  auto qout_off = 0;
  for (int e = 0; e < num_exps_incl_shared; e++) {
    auto exp_tokens = (e == num_experts) ? num_tokens : tokens_per_exp[e];
    if (exp_tokens == 0) continue;

    auto out_ld   = (e == num_experts) ? N * 2 * num_shared_experts : N * 2;
    auto pack_num = exp_qtag[e] == 0 ? 1 : (exp_qtag[e] == 1 ? 2 : 4);

    inp_list.push_back(
        torch::from_blob(inp_store.data_ptr<at::Half>() + exp_qinp_write_off[e], {qinp_sizes[e]}, options)
            .view({exp_tokens, K / pack_num}));
    if (qinp_scale_write_off > 0) {
      inp_scale_list.push_back(torch::from_blob(inp_scale_store.data_ptr<at::Half>() + exp_scale_write_off[e],
                                                {qinp_scale_sizes[e]}, options));
    } else {
      inp_scale_list.push_back(torch::empty({0}, options));
    }
    out_list.push_back(torch::from_blob(out_store.data_ptr<at::Half>() + qout_off, {exp_tokens * out_ld}, options)
                           .view({exp_tokens, out_ld}));
    qout_off += exp_tokens * out_ld;
  }

  return std::make_tuple(                                //
      inp_list, inp_scale_list, out_list,                //
      inp_store, inp_scale_store, out_store,             //
      recv_tokens_per_exp, perm_indices, dst_exp_sorted  //
  );
}

/// @return q_inp_list, q_inp_scale_list, q_inp, q_inp_scale, recv_tokens_pre_exp, perm_indices
auto silu_mul_then_quant(    //
    const Tensor& inp,       // [bs, seq_len, K]
    const int num_problems,  //
    const int num_experts,   //
    const int K,
    const int N,                   //
    const int num_shared_experts,  //
    const int num_tokens,          //
    const int topk,                //

    const Tensor& values_sorted,        //
    const Tensor& recv_tokens_per_exp,  //

    const std::vector<std::tuple<int, int, int, bool>>& qparams_per_exp  //
    ) -> std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, Tensor, Tensor, Tensor> {
  const bool has_shared_experts = num_shared_experts > 0;

  const int out_ld        = N;
  const int out_shared_ld = N * num_shared_experts;

  auto options = torch::TensorOptions().dtype(torch::kF16).device(torch::kCUDA);

  int alloc_tokens              = num_tokens * topk + (has_shared_experts ? num_tokens : 0);
  auto num_exps_incl_shared     = num_experts + (has_shared_experts ? 1 : 0);
  auto num_problems_incl_shared = num_problems + (has_shared_experts ? 1 : 0);

  auto cvt_qparams_to_tag = [](int qbits, int gsize) {
    if (qbits >= 16) {
      return 0;
    } else if (qbits == 8 && gsize == -1) {
      return 1;
    } else if (qbits == 4 && gsize == -1) {
      return 2;
    } else if (qbits == 4 && gsize == 128) {
      return 3;
    } else {
      return -1;
    }
  };

  auto host_buffer         = std::vector<int>(num_exps_incl_shared * 4);
  auto exp_qtag            = host_buffer.data();
  auto exp_st_token_idx    = host_buffer.data() + num_exps_incl_shared;
  auto exp_qinp_write_off  = host_buffer.data() + 2 * num_exps_incl_shared;
  auto exp_scale_write_off = host_buffer.data() + 3 * num_exps_incl_shared;

  int acc_tokens           = 0;
  int qinp_write_off       = 0;
  int qinp_scale_write_off = 0;

  auto qinp_sizes       = std::vector<int>(num_exps_incl_shared);
  auto qinp_scale_sizes = std::vector<int>(num_exps_incl_shared);

  auto tokens_per_exp = recv_tokens_per_exp.data_ptr<int64_t>();
  for (int e = 0; e < num_exps_incl_shared; e++) {
    auto cur_tag        = cvt_qparams_to_tag(std::get<0>(qparams_per_exp[e]), std::get<2>(qparams_per_exp[e]));
    auto inp_write_ld   = (e == num_experts) ? out_shared_ld : out_ld;
    auto cur_exp_tokens = (e == num_experts) ? num_tokens : tokens_per_exp[e];

    auto qinp_write_ld  = cur_tag == 0 ? inp_write_ld : (cur_tag == 1 ? inp_write_ld / 2 : inp_write_ld / 4);
    auto scale_write_ld = cur_tag == 0 ? 0 : ((cur_tag == 1 || cur_tag == 2) ? 1 : inp_write_ld / 128);

    exp_qtag[e]            = cur_tag;
    exp_st_token_idx[e]    = acc_tokens;
    exp_qinp_write_off[e]  = qinp_write_off;
    exp_scale_write_off[e] = qinp_scale_write_off;
    qinp_sizes[e]          = qinp_write_ld * cur_exp_tokens;
    qinp_scale_sizes[e]    = scale_write_ld * cur_exp_tokens;

    // std::cout << "exp: " << e << " qinp_size: " << qinp_sizes[e] << " qinp_scale_size: " << qinp_scale_sizes[e]
    //           << " inp_write_ld: " << inp_write_ld << " qinp: [" << cur_exp_tokens << ", " << qinp_write_ld << "]"
    //           << std::endl;

    qinp_write_off += qinp_write_ld * cur_exp_tokens;
    qinp_scale_write_off += scale_write_ld * cur_exp_tokens;
    acc_tokens += cur_exp_tokens;
  }

  size_t tot_bytes     = sizeof(int) * num_exps_incl_shared * 4;
  int* d_combined_buff = nullptr;
  checkCudaErrors(cudaMalloc(&d_combined_buff, tot_bytes));
  int offset      = 0;
  int* d_exp_qtag = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_st_token_idx = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_qinp_write_off = d_combined_buff + offset;
  offset += num_exps_incl_shared;
  int* d_exp_scale_write_off = d_combined_buff + offset;
  checkCudaErrors(cudaMemcpy(d_combined_buff, host_buffer.data(), tot_bytes, cudaMemcpyHostToDevice));

  auto inp_store       = torch::empty({qinp_write_off}, options);
  auto inp_scale_store = torch::empty({qinp_scale_write_off}, options);
  // for next kernel output (down_proj)
  auto out_store = torch::empty({(1 + topk) * num_tokens * K}, options);

  // auto block = std::min(K / 2, 1024);
  auto block = 1024;
  // determine necessary storage size:
  auto block_reduce_temp_bytes = sizeof(typename cub::BlockReduce<half, 1024>::TempStorage);
  // finally, we need to make sure that we can hold at least one half
  // needed in the kernel to exchange data after reduction
  auto smem_size = (std::max)(1 * sizeof(half), block_reduce_temp_bytes);

  auto tmp = torch::empty({alloc_tokens * out_shared_ld}, options);

  silu_mul_then_quant_kernel<<<alloc_tokens, block, smem_size>>>(         //
      (const half*)inp.data_ptr<at::Half>(),                              // inp
      (const half*)inp.data_ptr<at::Half>() + num_tokens * topk * N * 2,  // shared_experts inp
      (half*)inp_store.data_ptr<at::Half>(),                              // q_inp
      (half*)inp_scale_store.data_ptr<at::Half>(),                        // q_inp_scale
      values_sorted.data_ptr<int>(),                                      // determine which expert
      (half*)tmp.data_ptr<at::Half>(),                                    //
      d_exp_qtag,                                                         // quant tag for each expert
      d_exp_st_token_idx,                                                 // start token index for each expert inp
      d_exp_qinp_write_off,                                               // write offset for q_inp
      d_exp_scale_write_off,                                              //
      N,                                                                  //
      N * num_shared_experts,                                             //
      num_tokens * topk,                                                  //
      num_experts                                                         //
  );

  auto inp_list       = std::vector<Tensor>{};
  auto inp_scale_list = std::vector<Tensor>{};
  auto out_list       = std::vector<Tensor>{};
  inp_list.reserve(num_problems_incl_shared);
  inp_scale_list.reserve(num_problems_incl_shared);
  out_list.reserve(num_problems_incl_shared);

  checkCudaErrors(cudaFree(d_combined_buff));

  auto qout_off = 0;
  for (int e = 0; e < num_exps_incl_shared; e++) {
    auto exp_tokens = (e == num_experts) ? num_tokens : tokens_per_exp[e];
    if (exp_tokens == 0) continue;

    auto inp_ld_  = (e == num_experts) ? out_shared_ld : out_ld;
    auto pack_num = exp_qtag[e] == 0 ? 1 : (exp_qtag[e] == 1 ? 2 : 4);

    // std::cout << "exp: " << e << " qinp_size: " << qinp_sizes[e] << " qinp_scale_size: " << qinp_scale_sizes[e]
    //           << " tokens_per_exp: " << tokens_per_exp[e] << ", " << K / pack_num << std::endl;

    inp_list.push_back(
        torch::from_blob(inp_store.data_ptr<at::Half>() + exp_qinp_write_off[e], {qinp_sizes[e]}, options)
            .view({exp_tokens, inp_ld_ / pack_num}));
    if (qinp_scale_write_off > 0) {
      inp_scale_list.push_back(torch::from_blob(inp_scale_store.data_ptr<at::Half>() + exp_scale_write_off[e],
                                                {qinp_scale_sizes[e]}, options));
    } else {
      inp_scale_list.push_back(torch::empty({0}, options));
    }

    out_list.push_back(
        torch::from_blob(out_store.data_ptr<at::Half>() + qout_off, {exp_tokens * K}, options).view({exp_tokens, K}));
    qout_off += exp_tokens * K;
  }

  return std::make_tuple(                    //
      inp_list, inp_scale_list, out_list,    //
      inp_store, inp_scale_store, out_store  //
  );
}

PYBIND11_MODULE(mxmoe_ops, m) {
  m.def("gg_cutlass", &moe_cutlass);
  m.def("gg_cutlass_share_fused", &moe_cutlass_share_fused);
  m.def("gg_mxmoe_share_fused", &moe_mxmoe_share_fused);

  m.def("quant_inp_act", &quant_inp_act);
  m.def("silu_mul_then_quant", &silu_mul_then_quant);

  m.doc() = "A simple example python extension";
}
