#include <algorithm>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <thread>
#include <cuda_runtime.h>
#include <vector>

#include "cuda_utils.cuh"
#include "generated/hz_fused_decl.cuh"
// #include "generated/seq_launch_decl.cuh"
// #include "generated/ms_launch_decl.cuh"
#include "ref_impl.cuh"

#include "registry.cuh"
#include "test_utils.h"

using mxmoe::QParams;
using thrust::device_vector;
using thrust::host_vector;
using json = nlohmann::json;

using namespace tiled_gemm;

namespace fs = std::filesystem;

template <>
class fmt::formatter<GemmCoord> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(GemmCoord const& coord, Context& ctx) const {
    return format_to(ctx.out(), "({},{},{})", coord.m(), coord.n(), coord.k());
  }
};

template <>
class fmt::formatter<QParams> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(QParams const& p, Context& ctx) const {
    return format_to(ctx.out(), "w{}a{}_g{}_{}", p.qbits.y, p.qbits.x, p.gsize, p.sym ? "sym" : "asym");
  }
};

template <int WIDTH>
__host__ __device__ void print_bin(void* val) {
  static_assert(WIDTH % 4 == 0, "WIDTH must be a multiple of 4");
  auto pack = *reinterpret_cast<uint16_t*>(val);
  for (auto i = 0; i < WIDTH / 4; i++) {
    for (auto j = 0; j < 4; j++) {
      auto x = j + i * 4;
      printf("%d", (pack >> (WIDTH - x - 1)) & 1);
    }
    printf(" ");
  }
}

struct QShape {
  std::vector<int> shape{};
  int w_bits{16};
  int a_bits{16};
  int gsize{-1};
  bool sym{true};

  QShape() = default;

  // Serialize to JSON
  nlohmann::json to_json() const {
    return nlohmann::json{{"shape", shape}, {"w_bits", w_bits}, {"a_bits", a_bits}, {"gsize", gsize}, {"sym", sym}};
  }

  // Deserialize from JSON
  static QShape from_json(const nlohmann::json& j) {
    QShape qshape;
    qshape.shape  = j.at("shape").get<std::vector<int>>();
    qshape.w_bits = j.at("w_bits").get<int>();
    qshape.a_bits = j.at("a_bits").get<int>();
    qshape.gsize  = j.at("gsize").get<int>();
    qshape.sym    = j.at("sym").get<bool>();
    return qshape;
  }
};

template <>
class fmt::formatter<QShape> {
 public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
  template <typename Context>
  constexpr auto format(QShape const& p, Context& ctx) const {
    return format_to(ctx.out(), "(w{}a{}_g{}_{}, {})", p.w_bits, p.a_bits, p.gsize, p.sym ? "sym" : "asym", p.shape);
  }
};

namespace utils {

template <typename F, typename Res, typename Ref>
auto check_or_bench_group_gemm(                                  //
    CaseMode mode, std::string desc, bool is_gpu_kernel,         //
    const host_vector<GemmCoord>& problem_sizes,                 //
    F&& kernel, Res&& get_res, Ref&& get_ref,                    //
    bool verbose,                                                //
    int warm_up = 30, int bench_iter = 50, int sec_at_most = 30  //
) {
  double FLOPS = 0;
  int out_acc{};
  std::vector<int> prefixsum_out{};
  for (auto problem_size : problem_sizes) {
    auto M = problem_size.m();
    auto N = problem_size.n();
    auto K = problem_size.k();
    FLOPS += 2.0 * M * N * K;
    out_acc += M * N;
    prefixsum_out.push_back(out_acc);
  }
  auto get_coord = [&](int pos) {
    auto problem_idx = -1;
    for (auto i = 0; i < problem_sizes.size(); i++) {
      if (pos >= prefixsum_out[i]) {
        continue;
      }
      problem_idx = i;
      break;
    }
    // get coordinate in the problem
    pos -= problem_idx == 0 ? 0 : prefixsum_out[problem_idx - 1];
    auto coord = std::make_pair(pos / problem_sizes[problem_idx].n(), pos % problem_sizes[problem_idx].n());

    return std::make_pair(problem_idx, coord);
  };

  auto mode_str   = mode == CaseMode::CHECK ? "CHECK"sv : "BENCH"sv;
  auto print_info = fmt::format("\033[1;34m>>>\033[0m {} {}", mode_str, desc);

  if (mode == CaseMode::CHECK) {
    kernel();
    const auto& res = get_res();
    const auto& ref = get_ref();
    auto pos        = check_gemm(ref, res, print_info);
    if (pos != -1) {
      auto [pidx, coord] = get_coord(pos);
      fmt::println("pos: {}, problem: {}, coord: {}", pos, pidx, coord);
      if constexpr (std::is_integral_v<typename std::decay_t<decltype(ref)>::value_type>) {
        fmt::println("ref: {}, res: {}", ref[pos], res[pos]);
      } else {
        auto pld = problem_sizes[pidx].n();
        dump_arr(ref.data() + pos, pld, {8, 8}, "ref");
        dump_arr(res.data() + pos, pld, {8, 8}, "res");
      }
      throw ErrorInfo(fmt::format("Grouped GEMM correctness failed: {}", desc));
    }
    return std::make_pair(std::make_pair(0., 0.), desc);
  } else {
    auto [info, avg_time] = bench_func(desc, kernel, is_gpu_kernel, warm_up, bench_iter, sec_at_most);
    auto TFLOPS           = FLOPS / avg_time * 1e3 / 1e12;
    if (verbose) fmt::println("{}  TFLOPS: {}\n", info, TFLOPS);
    return std::make_pair(std::make_pair(double(avg_time), TFLOPS), desc);
  }
}

template <typename TA, typename TB, typename TC>
struct Input {
  host_vector<TA> h_As{};
  host_vector<TB> h_Bs{};
  host_vector<TC> h_Cs{};
  host_vector<TC> h_Refs{};
  device_vector<TA> As{};
  device_vector<TB> Bs{};
  device_vector<TC> Cs{};
  device_vector<TC> Refs{};

  host_vector<TA*> h_ptr_As{};
  host_vector<TB*> h_ptr_Bs{};
  host_vector<TC*> h_ptr_Cs{};
  host_vector<TC*> h_ptr_Ds{};
  device_vector<TA*> ptr_As{};
  device_vector<TB*> ptr_Bs{};
  device_vector<TC*> ptr_Cs{};
  device_vector<TC*> ptr_Ds{};

  device_vector<int32_t*> ptr_row_index{};

  host_vector<int64_t> h_ldas{};
  host_vector<int64_t> h_ldbs{};
  host_vector<int64_t> h_ldcs{};
  host_vector<int64_t> h_ldds{};
  device_vector<int64_t> ldas{};
  device_vector<int64_t> ldbs{};
  device_vector<int64_t> ldcs{};
  device_vector<int64_t> ldds{};

  host_vector<GemmCoord> h_problem_sizes{};
  device_vector<GemmCoord> problem_sizes{};

  host_vector<dim3> h_problem_sizes_dim3{};
  device_vector<dim3> problem_sizes_dim3{};
  int problem_count{};

  struct QInput {
    device_vector<half> pack_data_b{};
    device_vector<half> fq_data_b{};
    device_vector<TB*> pack_Bs{};
    device_vector<TB*> fq_Bs{};

    device_vector<half> pack_data_a{};
    device_vector<half> fq_data_a{};
    device_vector<TB*> pack_As{};
    device_vector<TB*> fq_As{};

    device_vector<half> scale_zp_data{};
    device_vector<half*> scale_zp_a{};
    device_vector<half*> scale_zp_b{};

    host_vector<QParams> qbits_list{};
    device_vector<QParams> d_qbits_list{};

    QInput() = default;

    static auto from_fp16(const device_vector<half*>& As, const device_vector<half*>& Bs,
                          const std::vector<QShape>& qshapes) {
      static_assert(std::is_same_v<TB, half>, "Only support half now");

      auto qinp = QInput{};

      int off_fq_a{}, off_pack_a{}, off_fq_b{}, off_pack_b{}, off_scale_zp{};
      for (auto i = 0; i < qshapes.size(); i++) {
        const auto a_bits = qshapes[i].a_bits;
        const auto w_bits = qshapes[i].w_bits;
        const auto gsize  = qshapes[i].gsize;
        const auto sym    = qshapes[i].sym;

        const bool is_quant = a_bits != 16 || w_bits != 16;

        const auto pack_num_a = 16 / a_bits;
        const auto pack_num_b = 16 / w_bits;

        auto M = qshapes[i].shape[0];
        auto N = qshapes[i].shape[1];
        auto K = qshapes[i].shape[2];

        const auto real_gsize = gsize == -1 ? K : gsize;

        // WxAx sym per-channel quantization
        if (w_bits < 16 && a_bits < 16 && sym) {
          if (w_bits != a_bits) {
            throw ErrorInfo(fmt::format("wbits`{}` abits`{}` should be equal in WxAx quant for now", w_bits, a_bits));
          }
          auto scale_a_size = sym ? (M * K / real_gsize) : (2 * M * K / real_gsize);
          auto scale_b_size = sym ? (N * K / real_gsize) : (2 * N * K / real_gsize);

          host_vector<half> h_pack_a, h_pack_b;
          auto d_qa      = device_vector<int>(M * K);
          auto d_fq_a    = device_vector<half>(M * K);
          auto d_scale_a = device_vector<half>(scale_a_size);
          auto d_qb      = device_vector<int>(N * K);
          auto d_fq_b    = device_vector<half>(N * K);
          auto d_scale_b = device_vector<half>(scale_b_size);

          quant_weight<<<M * K / real_gsize, 32>>>(        //
              As[i],                                       //
              d_fq_a.data().get(),                         //
              d_qa.data().get(),                           //
              d_scale_a.data().get(),                      //
              M * K / real_gsize, real_gsize, w_bits, sym  //
          );

          quant_weight<<<N * K / real_gsize, 32>>>(        //
              Bs[i],                                       //
              d_fq_b.data().get(),                         //
              d_qb.data().get(),                           //
              d_scale_b.data().get(),                      //
              N * K / real_gsize, real_gsize, a_bits, sym  //
          );

          auto qa = host_vector<int>(d_qa);
          auto qb = host_vector<int>(d_qb);
          if (a_bits == 4) {
            auto pack_data               = pack_wxax<nv_precision::s4>(qa, qb, M, N, K, a_bits);
            std::tie(h_pack_a, h_pack_b) = pack_data;
          } else if (a_bits == 8) {
            auto pack_data               = pack_wxax<int8_t>(qa, qb, M, N, K, a_bits);
            std::tie(h_pack_a, h_pack_b) = pack_data;
          }
          auto permuted_scale_a = permute_scale(host_vector<half>(d_scale_a), M, K, real_gsize, w_bits, sym);
          auto permuted_scale_b = permute_scale(host_vector<half>(d_scale_b), N, K, real_gsize, a_bits, sym);

          // dump_arr(h_pack_a.data(), K / pack_num_a, {M, K / pack_num_a}, "host packed A");
          // dump_arr<true>(h_pack_b.data(), K / pack_num_b, {K / pack_num_b, N}, "host packed B");
          // fmt::println("h_pack_a.size(): {}", h_pack_a.size());
          // fmt::println("h_pack_b.size(): {}", h_pack_b.size());

          auto cur_pack_a_size   = qinp.pack_data_a.size();
          auto cur_pack_b_size   = qinp.pack_data_b.size();
          auto cur_fq_a_size     = qinp.fq_data_a.size();
          auto cur_fq_b_size     = qinp.fq_data_b.size();
          auto cur_scale_zp_size = qinp.scale_zp_data.size();

          qinp.pack_data_a.resize(cur_pack_a_size + M * K / pack_num_a);
          qinp.pack_data_b.resize(cur_pack_b_size + N * K / pack_num_b);
          qinp.fq_data_a.resize(cur_fq_a_size + M * K);
          qinp.fq_data_b.resize(cur_fq_b_size + N * K);
          qinp.scale_zp_data.resize(cur_scale_zp_size + scale_a_size + scale_b_size);
          checkCudaErrors(cudaMemcpy((void*)(qinp.pack_data_a.data().get() + cur_pack_a_size),
                                     (const void*)(h_pack_a.data()), 2 * M * K / pack_num_a, cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.pack_data_b.data().get() + cur_pack_b_size),
                                     (const void*)(h_pack_b.data()), 2 * N * K / pack_num_b, cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.fq_data_a.data().get() + cur_fq_a_size),
                                     (const void*)(d_fq_a.data().get()), 2 * M * K, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.fq_data_b.data().get() + cur_fq_b_size),
                                     (const void*)(d_fq_b.data().get()), 2 * N * K, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.scale_zp_data.data().get() + cur_scale_zp_size),
                                     (const void*)(permuted_scale_a.data()), 2 * scale_a_size, cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.scale_zp_data.data().get() + cur_scale_zp_size + scale_a_size),
                                     (const void*)(permuted_scale_b.data()), 2 * scale_b_size, cudaMemcpyHostToDevice));

          off_pack_a += M * K / pack_num_a;
          off_pack_b += N * K / pack_num_b;
          off_fq_a += M * K;
          off_fq_b += N * K;
          off_scale_zp += scale_a_size + scale_b_size;
          continue;
        }

        // WxA16 per-channel quantizaiton
        if (w_bits < 16) {
          auto scale_zp_size = sym ? N * K / real_gsize : 2 * N * K / real_gsize;

          auto d_qb      = device_vector<int>(N * K);
          auto d_fq_b    = device_vector<half>(N * K);
          auto d_scale_b = device_vector<half>(scale_zp_size);
          quant_weight<<<N * K / real_gsize, 32>>>(        //
              Bs[i],                                       //
              d_fq_b.data().get(),                         //
              d_qb.data().get(),                           //
              d_scale_b.data().get(),                      //
              N * K / real_gsize, real_gsize, w_bits, sym  //
          );
          auto qb             = host_vector<int>(d_qb);
          auto permuted_scale = permute_scale(host_vector<half>(d_scale_b), N, K, real_gsize, w_bits, sym);
          auto permuted_b     = permute_weight(qb, N, K, w_bits, PermuteMode::Row);
          auto h_pack_b       = pack_weightonly(permuted_b, N, K, w_bits);

          auto cur_pack_a_size   = qinp.pack_data_a.size();
          auto cur_pack_b_size   = qinp.pack_data_b.size();
          auto cur_fq_a_size     = qinp.fq_data_a.size();
          auto cur_fq_b_size     = qinp.fq_data_b.size();
          auto cur_scale_zp_size = qinp.scale_zp_data.size();

          qinp.pack_data_a.resize(cur_pack_a_size + M * K / pack_num_a);
          qinp.pack_data_b.resize(cur_pack_b_size + N * K / pack_num_b);
          qinp.fq_data_a.resize(cur_fq_a_size + M * K);
          qinp.fq_data_b.resize(cur_fq_b_size + N * K);
          qinp.scale_zp_data.resize(cur_scale_zp_size + scale_zp_size);

          checkCudaErrors(cudaMemcpy((void*)(qinp.pack_data_a.data().get() + cur_pack_a_size), (const void*)(As[i]),
                                     2 * M * K / pack_num_a, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.fq_data_a.data().get() + cur_fq_a_size), (const void*)(As[i]),
                                     2 * M * K, cudaMemcpyDeviceToDevice));

          checkCudaErrors(cudaMemcpy((void*)(qinp.pack_data_b.data().get() + cur_pack_b_size),
                                     (const void*)(h_pack_b.data()), 2 * N * K / pack_num_b, cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy((void*)(qinp.fq_data_b.data().get() + cur_fq_b_size),
                                     (const void*)d_fq_b.data().get(), 2 * N * K, cudaMemcpyDeviceToDevice));

          checkCudaErrors(cudaMemcpy((void*)(qinp.scale_zp_data.data().get() + cur_scale_zp_size),
                                     (const void*)(permuted_scale.data()), scale_zp_size * 2, cudaMemcpyHostToDevice));

          // auto h_fq_a = host_vector<half>(qinp.pack_data_a);
          // auto h_fq_b = host_vector<half>(qinp.pack_data_b);

          // dump_arr(h_fq_a.data(), 256, {16, 256}, "host pack A");
          // dump_arr(h_fq_a.data(), 256, {16, 256}, "host pack B");

          off_pack_a += M * K / pack_num_a;
          off_pack_b += N * K / pack_num_b;
          off_fq_a += M * K;
          off_fq_b += N * K;
          off_scale_zp += scale_zp_size;
          continue;
        }

        // no quant
        if (!is_quant) {
          qinp.pack_data_a.resize(qinp.pack_data_a.size() + M * K);
          qinp.fq_data_a.resize(qinp.fq_data_a.size() + M * K);
          qinp.pack_data_b.resize(qinp.pack_data_b.size() + N * K);
          qinp.fq_data_b.resize(qinp.fq_data_b.size() + N * K);
          checkCudaErrors(cudaMemcpy(qinp.pack_data_a.data().get() + off_pack_a, As[i], M * K * 2, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy(qinp.fq_data_a.data().get() + off_fq_a, As[i], M * K * 2, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy(qinp.pack_data_b.data().get() + off_pack_b, Bs[i], N * K * 2, cudaMemcpyDeviceToDevice));
          checkCudaErrors(cudaMemcpy(qinp.fq_data_b.data().get() + off_fq_b, Bs[i], N * K * 2, cudaMemcpyDeviceToDevice));
          off_pack_a += M * K;
          off_pack_b += N * K;
          off_fq_a += M * K;
          off_fq_b += N * K;
        }
      }

      host_vector<QParams> qbits_list{};
      for (auto qshape : qshapes)
        qbits_list.push_back(QParams(make_int2(qshape.a_bits, qshape.w_bits), qshape.gsize, qshape.sym));

      qinp.qbits_list   = qbits_list;
      qinp.d_qbits_list = qbits_list;
      return qinp;
    }
  };

  QInput q_inp = QInput{};

  Input() = default;

  auto get_host_ptr(bool gather_load = false) {
    h_ptr_As.clear();
    h_ptr_Bs.clear();
    h_ptr_Cs.clear();
    h_ptr_Ds.clear();

    int off_a{}, off_b{}, off_c{};
    for (auto i = 0; i < problem_count; i++) {
      h_ptr_As.push_back(h_As.data() + off_a);
      h_ptr_Bs.push_back(h_Bs.data() + off_b);
      h_ptr_Cs.push_back(h_Cs.data() + off_c);
      h_ptr_Ds.push_back(h_Cs.data() + off_c);

      off_a += h_problem_sizes[i].m() * h_problem_sizes[i].k();
      off_b += h_problem_sizes[i].n() * h_problem_sizes[i].k();
      off_c += h_problem_sizes[i].m() * h_problem_sizes[i].n();
    }
    return std::make_tuple(h_ptr_As.data(), h_ptr_Bs.data(), h_ptr_Cs.data(), h_ptr_Ds.data(), h_ldas.data(),
                           h_ldbs.data(), h_ldcs.data(), h_ldds.data(), h_problem_sizes.data(), h_problem_sizes.data(),
                           problem_count);
  }

  auto get_device_ptr(bool gather_load = false) {
    ptr_As.clear();
    ptr_Bs.clear();
    ptr_Cs.clear();
    ptr_Ds.clear();

    int off_a{}, off_b{}, off_c{};
    for (auto i = 0; i < problem_count; i++) {
      ptr_As.push_back(As.data().get() + off_a);
      ptr_Bs.push_back(Bs.data().get() + off_b);
      ptr_Cs.push_back(Cs.data().get() + off_c);
      ptr_Ds.push_back(Cs.data().get() + off_c);

      off_a += h_problem_sizes[i].m() * h_problem_sizes[i].k();
      off_b += h_problem_sizes[i].n() * h_problem_sizes[i].k();
      off_c += h_problem_sizes[i].m() * h_problem_sizes[i].n();
    }
    return std::make_tuple(ptr_As.data().get(), ptr_Bs.data().get(), ptr_Cs.data().get(), ptr_Ds.data().get(),
                           ldas.data().get(), ldbs.data().get(), ldcs.data().get(), ldds.data().get(),
                           problem_sizes.data().get(), h_problem_sizes.data(), problem_count);
  }

  auto get_device_ref_ptr(bool gather_load = false) {
    ptr_As.clear();
    ptr_Bs.clear();
    ptr_Cs.clear();
    ptr_Ds.clear();

    int off_a{}, off_b{}, off_c{};
    for (auto i = 0; i < problem_count; i++) {
      ptr_As.push_back(As.data().get() + off_a);
      ptr_Bs.push_back(Bs.data().get() + off_b);
      ptr_Cs.push_back(Refs.data().get() + off_c);
      ptr_Ds.push_back(Refs.data().get() + off_c);

      off_a += h_problem_sizes[i].m() * h_problem_sizes[i].k();
      off_b += h_problem_sizes[i].n() * h_problem_sizes[i].k();
      off_c += h_problem_sizes[i].m() * h_problem_sizes[i].n();
    }
    return std::make_tuple(ptr_As.data().get(), ptr_Bs.data().get(), ptr_Cs.data().get(), ptr_Ds.data().get(),
                           ldas.data().get(), ldbs.data().get(), ldcs.data().get(), ldds.data().get(),
                           problem_sizes.data().get(), h_problem_sizes.data(), problem_count);
  }

  void _get_quant_ptr() {
    if (q_inp.qbits_list.size() == 0) {
      throw ErrorInfo(fmt::format("qbits list shouldn't be empty, got {}", q_inp.qbits_list.size()));
    }

    q_inp.pack_As.clear();
    q_inp.pack_Bs.clear();
    q_inp.scale_zp_a.clear();
    q_inp.scale_zp_b.clear();
    ptr_Cs.clear();
    ptr_Ds.clear();

    int off_a{}, off_b{}, off_c{}, off_scale_zp{};
    for (auto i = 0; i < problem_count; i++) {
      const auto M = h_problem_sizes[i].m();
      const auto N = h_problem_sizes[i].n();
      const auto K = h_problem_sizes[i].k();

      const auto a_bits       = q_inp.qbits_list[i].qbits.x;
      const auto w_bits       = q_inp.qbits_list[i].qbits.y;
      const auto sym          = q_inp.qbits_list[i].sym;
      const auto gsize        = q_inp.qbits_list[i].gsize;
      const auto real_gsize   = gsize == -1 ? K : gsize;
      const auto scale_size_a = a_bits >= 16 ? 0 : (sym ? M * K / real_gsize : 2 * M * K / real_gsize);
      const auto scale_size_b = w_bits >= 16 ? 0 : (sym ? N * K / real_gsize : 2 * N * K / real_gsize);

      const auto pack_num_a = 16 / a_bits;
      const auto pack_num_b = 16 / w_bits;

      ptr_Cs.push_back(Cs.data().get() + off_c);
      ptr_Ds.push_back(Cs.data().get() + off_c);

      // Each problem reads its packed segment, not the original FP16 bytes.
      q_inp.pack_As.push_back(q_inp.pack_data_a.data().get() + off_a);
      q_inp.pack_Bs.push_back(q_inp.pack_data_b.data().get() + off_b);
      q_inp.scale_zp_a.push_back(q_inp.scale_zp_data.data().get() + off_scale_zp);
      q_inp.scale_zp_b.push_back(q_inp.scale_zp_data.data().get() + off_scale_zp + scale_size_a);

      off_a += M * K / pack_num_a;
      off_b += N * K / pack_num_b;
      off_c += M * N;

      off_scale_zp += (scale_size_b + scale_size_a);
      // fmt::println("act_bits: {}, w_bits: {}, off_a: {}, off_b: {}, off_c: {}, off_scale_zp: {}", act_bits, w_bits,
      //              off_a, off_b, off_c, off_scale_zp);
    }
  }

  auto get_quant_ptr_dim3() {
    _get_quant_ptr();

    // fmt::println("{}", host_vector<QParams>(q_inp.d_qbits_list));
    // fmt::println("{}", q_inp.qbits_list);
    // printf("a_bits: %d, w_bits: %d, gsize: %d, sym: %d\n", q_inp.qbits_list[0].qbits.x, q_inp.qbits_list[0].qbits.y,
    //        q_inp.qbits_list[0].gsize, q_inp.qbits_list[0].sym);
    return std::make_tuple(                                            //
        q_inp.pack_As.data().get(),                                    //
        q_inp.pack_Bs.data().get(),                                    //
        q_inp.scale_zp_a.data().get(),                                 //
        q_inp.scale_zp_b.data().get(),                                 //
        ptr_Cs.data().get(), ptr_Ds.data().get(),                      //
        ldas.data().get(), ldbs.data().get(),                          //
        ldcs.data().get(), ldds.data().get(),                          //
        problem_sizes_dim3.data().get(), h_problem_sizes_dim3.data(),  //
        q_inp.d_qbits_list.data().get(), q_inp.qbits_list.data(),      //
        problem_count                                                  //
    );
  }

  auto get_quant_ptr() {
    _get_quant_ptr();
    return std::make_tuple(                                        //
        q_inp.pack_As.data().get(),                                //
        q_inp.pack_Bs.data().get(),                                //
        q_inp.scale_zp_a.data().get(),                             //
        q_inp.scale_zp_b.data().get(),                             //
        ptr_Cs.data().get(), ptr_Ds.data().get(),                  //
        ldas.data().get(), ldbs.data().get(),                      //
        ldcs.data().get(), ldds.data().get(),                      //
        problem_sizes.data().get(), h_problem_sizes.data(),        //
        q_inp.d_qbits_list.data().get(), q_inp.qbits_list.data(),  //
        problem_count                                              //
    );
  }

  auto get_fake_quant_ptr(bool gather_load = false) {
    if (q_inp.qbits_list.size() == 0) {
      throw ErrorInfo(fmt::format("qbits list shouldn't be empty, got {}", q_inp.qbits_list));
    }
    auto& inp = q_inp;

    inp.fq_As.clear();
    inp.fq_Bs.clear();
    ptr_Cs.clear();
    ptr_Ds.clear();

    int off_a{}, off_b{}, off_c{};
    for (auto i = 0; i < problem_count; i++) {
      const auto M = h_problem_sizes[i].m();
      const auto N = h_problem_sizes[i].n();
      const auto K = h_problem_sizes[i].k();

      ptr_Cs.push_back(Cs.data().get() + off_c);
      ptr_Ds.push_back(Cs.data().get() + off_c);
      // CUTLASS checks the same quantized values after dequantization to FP16.
      inp.fq_As.push_back(inp.fq_data_a.data().get() + off_a);
      inp.fq_Bs.push_back(inp.fq_data_b.data().get() + off_b);

      off_a += M * K;
      off_b += N * K;
      off_c += M * N;
    }
    return std::make_tuple(                                  //
        inp.fq_As.data().get(), inp.fq_Bs.data().get(),      //
        ptr_Cs.data().get(), ptr_Ds.data().get(),            //
        ldas.data().get(), ldbs.data().get(),                //
        ldcs.data().get(), ldds.data().get(),                //
        problem_sizes.data().get(), h_problem_sizes.data(),  //
        problem_count                                        //
    );
  }
};

auto parse_legacy_input(fs::path path) -> std::map<std::string, std::map<std::string, std::vector<QShape>>> {
  std::ifstream file(path);
  if (!file.is_open()) throw ErrorInfo(fmt::format("Failed to open input file: `{}`", path.string()));

  auto ret = std::map<std::string, std::map<std::string, std::vector<QShape>>>{
      {"test_case"s, std::map<std::string, std::vector<QShape>>{}},
  };

  constexpr auto avalibale_w_bits = std::array<int, 4>{16, 8, 4, 2};
  constexpr auto avalibale_a_bits = std::array<int, 3>{16, 8, 4};

  std::string line;
  int a_bits, w_bits, m, n, k, gsize;
  std::string sym;
  char x;

  std::vector<QShape> shapes;
  while (std::getline(file, line)) {
    auto qshape = QShape{};

    std::istringstream iss(line);
    iss >> a_bits >> x >> w_bits >> x >> gsize >> x >> sym >> m >> x >> n >> x >> k;

    if (std::all_of(begin(avalibale_w_bits), end(avalibale_w_bits), [&](auto b) { return w_bits != b; }))
      throw ErrorInfo(fmt::format("Ava w bits: {}. Got {}", avalibale_w_bits, w_bits));
    if (std::all_of(begin(avalibale_a_bits), end(avalibale_a_bits), [&](auto b) { return a_bits != b; }))
      throw ErrorInfo(fmt::format("Ava a bits: {}. Got {}", avalibale_a_bits, a_bits));

    qshape.a_bits = a_bits;
    qshape.w_bits = w_bits;
    qshape.gsize  = gsize;
    qshape.sym    = sym == "sym";
    qshape.shape  = {m, n, k};

    shapes.push_back(qshape);
  }
  ret["test_case"]["gate_up"] = shapes;

  return ret;
}

auto parse_json_input(fs::path filename) -> std::map<std::string, std::map<std::string, std::vector<QShape>>> {
  std::ifstream file(filename);
  if (!file.is_open()) throw ErrorInfo(fmt::format("Failed to open input json file: `{}`", filename.string()));

  json j;
  file >> j;

  std::map<std::string, std::map<std::string, std::vector<QShape>>> layers;
  // std::map<std::string, std::map<std::string, std::vector<QShape>>> data;
  for (auto& layer : j.items()) {
    if (layer.key() == "num_tokens") continue;  // 跳过非层项
    auto& layerData = layers[layer.key()];
    for (auto& item : layer.value().items()) {
      std::vector<QShape> shapes;
      for (const auto& shapeJson : item.value()) {
        shapes.push_back(QShape::from_json(shapeJson));
      }
      layerData[item.key()] = shapes;
    }
  }

  return layers;
}

auto parse_input_file(fs::path path) {
  // host_vector<GemmCoord> problem_sizes, const host_vector<QParams>& qbits_list, int seed = 42;

  if (!fs::exists(path)) throw ErrorInfo(fmt::format("input file: `{}` not found", path.string()));

  fmt::println(">>> Parsing input file: `{}`", path.string());

  std::map<std::string, std::vector<std::vector<QShape>>> layer_inputs;
  if (path.extension() == ".json") {
    return parse_json_input(path);
  } else {
    return parse_legacy_input(path);
  }
}

using BenchResult = std::pair<std::pair<double, double>, std::string>;

template <typename TA, typename TB, typename TC>
auto bench_one_case(                  //
    Input<TA, TB, TC>& input,         //
    CaseMode mode, bool verbose,      //
    std::vector<std::string> filter,  //
    std::string out_file = ""s        //
    ) -> std::map<std::string, BenchResult> {
  using Minfo = RCR_FP16FP16FP16;

  auto bench_results = std::map<std::string, std::vector<BenchResult>>{
      {"cutlass", std::vector<BenchResult>{}},    //
      {"base-impl", std::vector<BenchResult>{}},  //
      {"mxmoe", std::vector<BenchResult>{}},      //
  };

  auto check_or_bench = [&](auto& _input, auto& kernel, auto kernel_name, auto&& params) {
    thrust::fill(_input.h_Cs.begin(), _input.h_Cs.end(), 0);
    thrust::fill(_input.Cs.begin(), _input.Cs.end(), 0);
    auto bench_res = utils::check_or_bench_group_gemm(
        mode, kernel_name, true, _input.h_problem_sizes,  //
        [&] { std::apply(kernel, params); },              //
        [&]() -> const host_vector<TC>& {
          _input.h_Cs = _input.Cs;
          return _input.h_Cs;
        },                                                          //
        [&]() -> const host_vector<TC>& { return _input.h_Refs; },  //
        verbose, 20, 50                                             //
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return bench_res;
  };

  if (mode == CaseMode::CHECK) {
    std::apply(mxmoe_ref::group_gemm_cutlass_f16<Minfo, TileConfig<128, 128, 64, 1, 2, 1, 3>>,
               input.get_fake_quant_ptr());
    input.h_Refs = input.Cs;
  }

#if COMPUTE_ARCH == 80 || COMPUTE_ARCH == 86
  auto fp16_tile_cfgs = std::make_tuple(       //
      TileConfig<128, 256, 32, 2, 4, 1, 3>{},  //
#if COMPUTE_ARCH == 80
      // These two candidates exceed SM86's per-block shared-memory capacity.
      TileConfig<128, 256, 64, 2, 4, 1, 3>{},  //

      TileConfig<128, 128, 64, 2, 2, 1, 4>{},  //
#endif
      TileConfig<128, 128, 64, 2, 2, 1, 3>{},  //
      TileConfig<128, 128, 32, 2, 2, 1, 3>{},  //

      TileConfig<64, 128, 64, 2, 2, 1, 4>{},  //
      TileConfig<64, 128, 64, 2, 2, 1, 3>{},  //
      TileConfig<64, 128, 32, 2, 2, 1, 3>{}   //

      //  TileConfig<64, 128, 64, 1, 2, 1, 4>{},  //
      //  TileConfig<32, 128, 64, 1, 2, 1, 4>{},  //

      //  TileConfig<16, 128, 64, 1, 2, 2, 4>{},  //
      //  TileConfig<16, 256, 64, 1, 4, 1, 4>{}   //
  );
#elif COMPUTE_ARCH == 89
  auto fp16_tile_cfgs = std::make_tuple(  //

      TileConfig<128, 256, 32, 2, 4, 1, 3>{}  //

      // TileConfig<128, 256, 32, 2, 4, 1, 3>{},  //

      // TileConfig<128, 128, 64, 2, 2, 1, 3>{},  //
      // TileConfig<128, 128, 32, 2, 2, 1, 3>{},  //

      // TileConfig<64, 128, 64, 2, 2, 1, 4>{},  //
      // TileConfig<64, 128, 64, 2, 2, 1, 3>{},  //
      // TileConfig<64, 128, 32, 2, 2, 1, 3>{},  //

      // TileConfig<64, 256, 32, 1, 4, 1, 4>{},  //
      // TileConfig<64, 128, 64, 1, 2, 1, 4>{},  //
      // TileConfig<32, 128, 64, 1, 2, 1, 4>{},  //
      // TileConfig<16, 128, 64, 1, 2, 2, 4>{}   //
      //  TileConfig<16, 256, 64, 1, 4, 1, 4>{}   //
  );
#else
#endif

  // cutlass
  {
    auto run = [&](auto t) {
      using Tile  = decltype(t);
      auto kernel = mxmoe_ref::group_gemm_cutlass_f16<Minfo, Tile>;
      return check_or_bench(input, kernel, tag_name<Tile>("cutlass"), input.get_fake_quant_ptr());
    };
    std::apply([&](auto... t) { (bench_results["cutlass"].push_back(run(t)), ...); }, fp16_tile_cfgs);
  }

  // // base-impl
  // {
  //   auto run = [&](auto t) {
  //     using Tile  = decltype(t);
  //     auto kernel = mxmoe_ref::my_group_gemm<RCR_FP16FP16FP16, Tile>;
  //     return check_or_bench(input, kernel, tag_name<Tile>("base-impl"), input.get_fake_quant_ptr());
  //   };
  //   std::apply([&](auto... t) { (bench_results["base-impl"].push_back(run(t)), ...); }, fp16_tile_cfgs);
  // }
  // mxmoe
  {
    int num_kernels = mxmoe::GetGlobalRegistry().storage_.size();
    fmt::println("num of kernel table entry: {}", num_kernels);

    for (auto i = 0; i < num_kernels; i++) {
      auto kernel_desc = mxmoe::GetGlobalRegistry()[i];
      auto kernel_name = kernel_desc.cfg_str;
      bool skip        = false;
      if (!filter.empty()) {
        skip = true;
        for (auto& f : filter) {
          if (kernel_name.find(f) != std::string::npos) {
            skip = false;
            break;
          }
        }
      }
      if (skip) continue;
      auto kernel = kernel_desc.func;
      fmt::println("{}", kernel_name);
      bench_results["mxmoe"].push_back(check_or_bench(input, kernel, kernel_name, input.get_quant_ptr_dim3()));
    }
  }

  auto get_best_res = [](std::vector<BenchResult>& res_vec) {
    if (res_vec.empty()) return BenchResult{{0, 0}, "N/A"};
    std::sort(begin(res_vec), end(res_vec), [](auto& a, auto& b) { return a.first.first < b.first.first; });
    auto best_res = res_vec[0];
    return best_res;
  };

  std::vector<std::pair<std::pair<double, double>, std::string>> mxmoe_seq;
  std::vector<std::pair<std::pair<double, double>, std::string>> mxmoe_hz;
  std::vector<std::pair<std::pair<double, double>, std::string>> mxmoe_ms;
  for (const auto& item : bench_results["mxmoe"]) {
    const std::string& str = item.second;
    if (str.find("seq_launch") != std::string::npos) {
      mxmoe_seq.push_back(item);
    } else if (str.find("hz_fused") != std::string::npos) {
      mxmoe_hz.push_back(item);
    } else if (str.find("ms_launch") != std::string::npos) {
      mxmoe_ms.push_back(item);
    }
  }

  auto best_results = std::map<std::string, BenchResult>{
      {"cutlass", get_best_res(bench_results["cutlass"])},      //
      {"base-impl", get_best_res(bench_results["base-impl"])},  //
      {"mxmoe-seq", get_best_res(mxmoe_seq)},                   //
      {"mxmoe-ms", get_best_res(mxmoe_ms)},                     //
      {"mxmoe-hz", get_best_res(mxmoe_hz)},                     //
  };

  auto TFLOPS_baseline = best_results["cutlass"].first.second;
  // fmt::println("problem_sizes[{}]: {}", input.problem_count, shapes);
  // fmt::println("qbits_list: {}", input.q_inp.qbits_list);
  for (auto& [impl, res] : best_results) {
    auto [bench_info, kernel_name] = res;
    auto [avg_time, TFLOPS]        = bench_info;
    auto speedup                   = TFLOPS / TFLOPS_baseline;
    fmt::println("{}:\navg_time: {:<8.4f}, TFLOPS: {:<9.4f}, speedup: {:<.3f}", kernel_name, avg_time, TFLOPS, speedup);
  }
  if (!out_file.empty()) {
    std::ofstream file(out_file);
    if (!file.is_open()) throw ErrorInfo(fmt::format("Failed to open output file: `{}`", out_file));
    file << fmt::format("kernel_name,avg_time,TFLOPS,speedup\n");

    for (auto& [bench_info, kernel_name] : mxmoe_hz) {
      auto [avg_time, TFLOPS] = bench_info;
      auto speedup            = TFLOPS / TFLOPS_baseline;
      file << fmt::format("\"{}\",{:.4f},{:.4f},{:.3f}\n", kernel_name, avg_time, TFLOPS, speedup);
    }
  }

  return best_results;
}

template <typename TA, typename TB, typename TC>
auto bench_input_file(fs::path filename, CaseMode mode, bool verbose, std::vector<std::string> filter,
                      std::string out_file, int seed = 42) {
  using InputType = Input<TA, TB, TC>;

  auto input_shapes = parse_input_file(filename);

  auto rng = std::default_random_engine(seed);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);

  for (const auto& [case_name, case_shapes] : input_shapes) {
    for (const auto& [gg_name, qshapes] : case_shapes) {
      auto bench_name = fmt::format("{}-{}", case_name, gg_name);
      fmt::println(">>> Initializing Full-Precision Weight of `{}`", bench_name);

      auto input = InputType{};
      size_t size_As{}, size_Bs{}, size_Cs{};
      for (auto problem_size : qshapes) {
        auto M = problem_size.shape[0];
        auto N = problem_size.shape[1];
        auto K = problem_size.shape[2];
        size_As += M * K;
        size_Bs += N * K;
        size_Cs += M * N;

        input.problem_sizes.push_back({M, N, K});
        input.h_problem_sizes.push_back({M, N, K});

        input.h_ldas.push_back(K);
        input.h_ldbs.push_back(K);
        input.h_ldcs.push_back(N);
        input.h_ldds.push_back(N);
        input.ldas.push_back(K);
        input.ldbs.push_back(K);
        input.ldcs.push_back(N);
        input.ldds.push_back(N);
      }
      input.problem_count = int(qshapes.size());
      input.h_As.resize(size_As);
      input.h_Bs.resize(size_Bs);
      input.h_Cs.resize(size_Cs);

      // std::fill(begin(input.h_As), end(input.h_As), TA(1));
      // std::fill(begin(input.h_Bs), end(input.h_Bs), TB(1));
      std::generate(begin(input.h_As), end(input.h_As), [&dis, &rng]() { return TA(dis(rng)); });
      std::generate(begin(input.h_Bs), end(input.h_Bs), [&dis, &rng]() { return TB(dis(rng)); });
      std::fill(begin(input.h_Cs), end(input.h_Cs), TC(0));
      input.h_Refs = input.h_Cs;
      input.As     = input.h_As;
      input.Bs     = input.h_Bs;
      input.Cs     = input.h_Cs;
      input.Refs   = input.Cs;

      input.h_problem_sizes_dim3.resize(input.h_problem_sizes.size());
      std::transform(begin(input.h_problem_sizes), end(input.h_problem_sizes), begin(input.h_problem_sizes_dim3),
                     [](auto& p) { return dim3(p.m(), p.n(), p.k()); });
      input.problem_sizes_dim3 = input.h_problem_sizes_dim3;

      fmt::println("Initializing Quantized Weight: {}", qshapes);
      input.get_device_ptr();
      auto qinp   = InputType::QInput::from_fp16(input.ptr_As, input.ptr_Bs, qshapes);

      // input.As.clear();
      // input.As.shrink_to_fit();
      // input.Bs.clear();
      // input.Bs.shrink_to_fit();
      input.Refs.clear();
      input.Refs.shrink_to_fit();

      input.q_inp = qinp;
      fmt::println("Weight initialized!");

      auto cur_out_file_name = out_file.empty() ? ""s : fmt::format("{}-{}.csv", out_file, bench_name);
      auto bench_results = bench_one_case(input, mode, verbose, filter, cur_out_file_name);
    }
  }
}

// Only operand preparation differs: INT4 uses packed bytes, INT8 uses signed bytes.
// Checking and timing go through the same helpers as the F16/MxMoE path.
template <int Bits>
void run_cutlass_integer_group(const std::string& name, const std::vector<IntegerGemmProblem>& problems,
                               CaseMode mode) {
  static_assert(Bits == 4 || Bits == 8);
  using Element = std::conditional_t<Bits == 4, cutlass::int4b_t, int8_t>;
  if (problems.empty()) throw ErrorInfo(fmt::format("{} has no problems", name));
  const int count = static_cast<int>(problems.size());
  host_vector<GemmCoord> host_problems(count);
  host_vector<int64_t> host_k(count), host_n(count);
  std::vector<size_t> offsets(count + 1, 0);
  long long flops = 0;
  for (int i = 0; i < count; ++i) {
    const auto& p = problems[i];
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) {
      throw ErrorInfo(fmt::format("{} has unsupported shape ({}, {}, {}); empty experts are not in this path",
                                  name, p.m, p.n, p.k));
    }
    if (p.k % (128 / Bits) != 0)
      throw ErrorInfo(fmt::format("{} K={} is not a multiple of {}", name, p.k, 128 / Bits));
    if (p.n % 4 != 0) throw ErrorInfo(fmt::format("{} N={} is not a multiple of 4", name, p.n));
    host_problems[i] = GemmCoord(p.m, p.n, p.k);
    host_k[i] = p.k;
    host_n[i] = p.n;
    offsets[i + 1] = offsets[i] + size_t(p.m) * p.n;
    flops += 2LL * p.m * p.n * p.k;
  }

  std::vector<std::vector<int8_t>> host_a(count), host_b(count);
  std::vector<cutlass::DeviceAllocation<Element>> a(count), b(count);
  device_vector<int32_t> output(offsets.back(), int32_t{0x7f7f7f7f});
  host_vector<Element*> host_ptr_a(count), host_ptr_b(count);
  host_vector<int32_t*> host_ptr_c(count);
  for (int i = 0; i < count; ++i) {
    if constexpr (Bits == 4) fill_w4a4(i, problems[i], host_a[i], host_b[i]);
    else fill_w8a8(problems[i], host_a[i], host_b[i]);
    // CUTLASS allocations count logical elements, including for sub-byte types.
    a[i].reset(host_a[i].size());
    b[i].reset(host_b[i].size());
    if constexpr (Bits == 4) {
      auto packed_a = pack_cutlass_int4(host_a[i]);
      auto packed_b = pack_cutlass_int4(host_b[i]);
      a[i].copy_from_host(reinterpret_cast<const Element*>(packed_a.data()));
      b[i].copy_from_host(reinterpret_cast<const Element*>(packed_b.data()));
    } else {
      a[i].copy_from_host(host_a[i].data());
      b[i].copy_from_host(host_b[i].data());
    }
    host_ptr_a[i] = a[i].get();
    host_ptr_b[i] = b[i].get();
    host_ptr_c[i] = output.data().get() + offsets[i];
  }
  device_vector<Element*> ptr_a = host_ptr_a, ptr_b = host_ptr_b;
  device_vector<int32_t*> ptr_c = host_ptr_c;
  device_vector<int64_t> ld_k = host_k, ld_n = host_n;
  device_vector<GemmCoord> sizes = host_problems;
  constexpr auto kernel = [] {
    if constexpr (Bits == 4) return mxmoe_ref::group_gemm_cutlass_w4a4;
    else return mxmoe_ref::group_gemm_cutlass_w8a8;
  }();
  auto launch = [&] {
    kernel(ptr_a.data().get(), ptr_b.data().get(), ptr_c.data().get(), ptr_c.data().get(),
           ld_k.data().get(), ld_k.data().get(), ld_n.data().get(), ld_n.data().get(),
           sizes.data().get(), host_problems.data(), count);
  };
  host_vector<int32_t> result, reference;
  auto get_result = [&]() -> const host_vector<int32_t>& {
    result = output;
    return result;
  };
  auto get_reference = [&]() -> const host_vector<int32_t>& {
    reference.resize(offsets.back());
    for (int i = 0; i < count; ++i) {
      const auto& p = problems[i];
      for (int row = 0; row < p.m; ++row) {
        for (int col = 0; col < p.n; ++col) {
          reference[offsets[i] + size_t(row) * p.n + col] = reference_integer_dot(host_a[i], host_b[i], p.k, row, col);
        }
      }
    }
    return reference;
  };
  // Keep the existing log field for consumers; it counts 2*M*N*K, not MACs.
  fmt::println(">>> {} problems={} macs={}", name, count, flops);
  auto measured = check_or_bench_group_gemm(mode, name, true, host_problems, launch, get_result, get_reference,
                                           false, 30, 50, std::numeric_limits<int>::max());
  if (mode == CaseMode::BENCH) {
    fmt::println("  median_ms={:.4f} timing_scope=grouped_call output=int32 dequant=false", measured.first.first);
  } else {
    for (int i = 0; i < count; ++i) {
      const auto& p = problems[i];
      if (p.inexact_accumulator) {
        if (result[offsets[i]] != 33032065) throw ErrorInfo("FP32-inexact accumulator check failed");
        fmt::println("  problem {} ({},{},{}) exact, accumulator {} is not FP32-representable",
                     i, p.m, p.n, p.k, result[offsets[i]]);
      } else {
        fmt::println("  problem {} ({},{},{}) exact INT32", i, p.m, p.n, p.k);
      }
    }
  }
}

template <int Bits>
int run_cutlass_integer(CaseMode mode, const std::string& input) {
  const bool check = mode == CaseMode::CHECK;
  fmt::println("CUTLASS W{0}A{0}: standard GemmGrouped, INT{0} A/B, INT32 accumulate and store, no dequant", Bits);
  fmt::println("tile 64x128x{}, warp 32x64x{}, mma 16x8x{}, stages 3, alignment {}",
               1024 / Bits, 1024 / Bits, 256 / Bits, 128 / Bits);
  if (check || input.empty()) {
    run_cutlass_integer_group<Bits>("builtin",
        {{1, 16, Bits == 8 ? 2048 : 64, Bits == 8}, {17, 32, 64, false}, {33, 16, 32, false},
         {8, 16, Bits == 8 ? 32 : 64, false}, {64, 128, 128, false}}, mode);
  }
  if (input.empty()) return 0;

  auto layers = parse_input_file(input);
  int validated = 0, skipped = 0;
  for (const auto& [case_name, groups] : layers) {
    for (const auto& [group_name, shapes] : groups) {
      std::vector<IntegerGemmProblem> problems;
      long long flops = 0;
      for (const auto& shape : shapes) {
        if (shape.shape.size() != 3) throw ErrorInfo(fmt::format("{} shape is not M,N,K", shape));
        if (shape.a_bits != Bits || shape.w_bits != Bits || !shape.sym || shape.gsize != -1) {
          throw ErrorInfo(fmt::format("{}-{} is not symmetric W{}A{} gsize=-1: {}", case_name, group_name, Bits, Bits, shape));
        }
        problems.push_back({shape.shape[0], shape.shape[1], shape.shape[2], false});
        flops += 2LL * shape.shape[0] * shape.shape[1] * shape.shape[2];
      }
      auto name = fmt::format("{}-{}", case_name, group_name);
      if (check && flops > 20000000) {
        std::cerr << fmt::format("{} exceeds the 20-million-operation host reference limit; this input is not validated.\n", name);
        ++skipped;
        continue;
      }
      run_cutlass_integer_group<Bits>(name, problems, mode);
      ++validated;
    }
  }
  if (check && (skipped > 0 || validated == 0)) {
    std::cerr << fmt::format("input was not fully validated: validated={} skipped={}\n", validated, skipped);
    return 1;
  }
  return 0;
}

}  // namespace utils

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("Benchmark Program", "1.0");
  program.add_argument("mode").help("Mode of operation: bench or check").choices("bench", "check");
  program.add_argument("--verbose")
      .default_value(false)
      .implicit_value(true)
      .help("Enable verbose mode for extra logging");
  program.add_argument("--backend")
      .default_value("default"s)
      .choices("default", "cutlass_w8a8", "cutlass_w4a4")
      .help("default keeps the existing path; cutlass_w8a8 and cutlass_w4a4 run standard integer grouped GEMM");
  program.add_argument("--input").default_value(""s).help("Path to the input file. Required for the default backend");
  program.add_argument("--output").default_value(""s).help("Path to the save the output");
  program.add_argument("--filter")
      .default_value(std::vector<std::string>())
      .nargs(argparse::nargs_pattern::any)
      .help("Filter strings to apply during test");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  auto verbose   = program.get<bool>("--verbose");
  auto workload  = program.get<std::string>("--input");
  auto out_file  = program.get<std::string>("--output");
  auto mode      = program.get<std::string>("mode");
  auto case_mode = mode == "check" ? CaseMode::CHECK : CaseMode::BENCH;
  auto filter    = program.get<std::vector<std::string>>("--filter");
  fmt::println("filter: {}", filter);
  auto backend   = program.get<std::string>("--backend");
  if (backend == "cutlass_w8a8") return utils::run_cutlass_integer<8>(case_mode, workload);
  if (backend == "cutlass_w4a4") return utils::run_cutlass_integer<4>(case_mode, workload);
  if (workload.empty()) {
    std::cerr << "--input is required unless --backend cutlass_w8a8 or cutlass_w4a4" << std::endl;
    std::cerr << program;
    return 1;
  }

  using TA = half;
  using TB = half;
  using TC = half;

  utils::bench_input_file<TA, TB, TC>(workload, case_mode, verbose, filter, out_file);
}