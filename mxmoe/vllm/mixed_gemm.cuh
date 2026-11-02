#pragma once

// Mixed A4/A8/A16 grouped GEMM support for kernels.cu. Include only after the
// legacy helpers, ScratchLayout, TileW2/W4/W8, and Torch/CUDA headers.

namespace mxmoe_mixed {

constexpr int kMixedBM = 64;
constexpr int kMixedBN = 128;
constexpr int kMixedBK = 128;

using QA4 = tiled_gemm::QConfig<half, true, 4, -1, tiled_gemm::PackDim::K>;
using QB4 = tiled_gemm::QConfig<half, true, 4, 128, tiled_gemm::PackDim::K>;
using QA8 = tiled_gemm::QConfig<half, true, 8, -1, tiled_gemm::PackDim::K>;
using QB8 = tiled_gemm::QConfig<half, true, 8, 128, tiled_gemm::PackDim::K>;
using TileA4 = tiled_gemm::TileConfig<kMixedBM, kMixedBN, kMixedBK, 1, 4, 1, 1, -1,
                                      tiled_gemm::MMA_S4_K64, QA4, QB4>;
using TileA8 = tiled_gemm::TileConfig<kMixedBM, kMixedBN, kMixedBK, 1, 4, 1, 1, -1,
                                      tiled_gemm::MMA_S8_K32, QA8, QB8>;

template <typename TileCfg>
struct LowPlan {
  using TileA = tiled_gemm::QuantTileA<half, tiled_gemm::Layout::RowMajor, TileCfg,
                                      typename TileCfg::QConfigA>;
  using TileB = tiled_gemm::QuantTileB<half, tiled_gemm::Layout::ColMajor, TileCfg,
                                      typename TileCfg::QConfigB>;
  static constexpr int bytes_a = TileA::VEC_PER_ROW * TileA::TILE_ROW * 16;
  static constexpr int bytes_b = TileB::VEC_PER_ROW * TileB::TILE_ROW * 16;
  static constexpr int off_sa = bytes_a + bytes_b;
  static constexpr int off_sum = off_sa + kMixedBM * static_cast<int>(sizeof(float));
  static constexpr int off_sb = off_sum + kMixedBM * static_cast<int>(sizeof(int32_t));
  static constexpr int off_zb = off_sb + kMixedBN * static_cast<int>(sizeof(half));
  static constexpr int total = off_zb + kMixedBN * static_cast<int>(sizeof(half));
  static constexpr int warps = TileCfg::WARP_M * TileCfg::WARP_N * TileCfg::WARP_K;
};

template <typename TileCfg>
struct StridedQuantTileA
    : tiled_gemm::QuantTileA<half, tiled_gemm::Layout::RowMajor, TileCfg,
                             typename TileCfg::QConfigA> {
  using Base = tiled_gemm::QuantTileA<half, tiled_gemm::Layout::RowMajor, TileCfg,
                                      typename TileCfg::QConfigA>;
  __device__ StridedQuantTileA(const half* a, int m, int k, int packed_stride,
                               int bx, int by, int warp_y, int warp_z, int tidx)
      : Base(a, m, k, bx, by, warp_y, warp_z, tidx) {
    this->leading = packed_stride;
    this->cta_A = a + static_cast<int64_t>(by) * Base::BM * packed_stride;
  }
};

template <typename TileCfg>
struct SlicedTileC
    : tiled_gemm::GlobalTileC<half, tiled_gemm::Layout::RowMajor, TileCfg> {
  using Base = tiled_gemm::GlobalTileC<half, tiled_gemm::Layout::RowMajor, TileCfg>;
  static constexpr int WFM = TileCfg::WARP_FRAG_M;
  static constexpr int WFN = TileCfg::WARP_FRAG_N;
  __device__ SlicedTileC(half* c, int m, int n, int output_ld, int bx, int by, int tidx)
      : Base(c, m, output_ld, bx, by, tidx) {
    this->leading = output_ld;
    this->cta_MAX_LOGIC_ROW = m - by * TileCfg::BM;
    this->cta_MAX_LOGIC_COL = n - bx * TileCfg::BN;
    this->cta_C = c + static_cast<int64_t>(by) * TileCfg::BM * output_ld + bx * TileCfg::BN;
  }

  __device__ __forceinline__ void load_mixed_params(
      float sa[WFM * 2], int32_t suma[WFM * 2], half sb[WFN * 2], half zb[WFN * 2],
      const float* sm_sa, const int32_t* sm_sum, const half* sm_sb, const half* sm_zb,
      int lane, int warp_x, int warp_y) {
    const int row = lane / 4;
    const int col = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < WFM; ++i) {
      const int base = warp_y * TileCfg::WARP_TILE_M + i * TileCfg::mmaM + row;
      sa[2 * i] = sm_sa[base];
      sa[2 * i + 1] = sm_sa[base + 8];
      suma[2 * i] = sm_sum[base];
      suma[2 * i + 1] = sm_sum[base + 8];
    }
#pragma unroll
    for (int j = 0; j < WFN; ++j) {
      const int base = warp_x * TileCfg::WARP_TILE_N + j * TileCfg::mmaN + col;
      sb[2 * j] = sm_sb[base];
      sb[2 * j + 1] = sm_sb[base + 1];
      zb[2 * j] = sm_zb[base];
      zb[2 * j + 1] = sm_zb[base + 1];
    }
  }

  template <int CENTER>
  __device__ __forceinline__ void scale_and_correct(
      const float sa[WFM * 2], const int32_t suma[WFM * 2],
      const half sb[WFN * 2], const half zb[WFN * 2],
      int32_t input[WFM * WFN][TileCfg::FRAG_C_TH],
      float output[WFM * WFN][TileCfg::FRAG_C_TH]) {
#pragma unroll
    for (int i = 0; i < WFM; ++i) {
#pragma unroll
      for (int j = 0; j < WFN; ++j) {
        const int f = i * WFN + j;
        const int corr00 = suma[2 * i] * (CENTER - __half2int_rn(zb[2 * j]));
        const int corr01 = suma[2 * i] * (CENTER - __half2int_rn(zb[2 * j + 1]));
        const int corr10 = suma[2 * i + 1] * (CENTER - __half2int_rn(zb[2 * j]));
        const int corr11 = suma[2 * i + 1] * (CENTER - __half2int_rn(zb[2 * j + 1]));
        output[f][0] += static_cast<float>(input[f][0] + corr00) * sa[2 * i] * __half2float(sb[2 * j]);
        output[f][1] += static_cast<float>(input[f][1] + corr01) * sa[2 * i] * __half2float(sb[2 * j + 1]);
        output[f][2] += static_cast<float>(input[f][2] + corr10) * sa[2 * i + 1] * __half2float(sb[2 * j]);
        output[f][3] += static_cast<float>(input[f][3] + corr11) * sa[2 * i + 1] * __half2float(sb[2 * j + 1]);
      }
    }
  }
};

template <typename TileCfg, int WEIGHT_BITS>
__device__ __forceinline__ void lowbit_body(
    uint8_t* smem, half** ptr_a, half** ptr_b, half** ptr_scale, half** ptr_c,
    float** ptr_as, int32_t** ptr_sum, dim3* sizes, int* prefix, int problems,
    int packed_stride, int output_ld) {
  using Plan = LowPlan<TileCfg>;
  using TileA = StridedQuantTileA<TileCfg>;
  using TileB = typename Plan::TileB;
  using TileC = SlicedTileC<TileCfg>;
  using FragA = typename TileA::FragA;
  using FragB = typename TileB::FragB;
  using FragC = typename TileC::FragC;
  constexpr int CENTER = 1 << (WEIGHT_BITS - 1);
  const int lane = threadIdx.x;
  const int tidx = lane + threadIdx.y * WarpSize;
  const int wx = TileCfg::warp_x(), wy = TileCfg::warp_y(), wz = TileCfg::warp_z();
  auto* sm_a = reinterpret_cast<uint4*>(smem);
  auto* sm_b = reinterpret_cast<uint4*>(smem + Plan::bytes_a);
  auto* sm_sa = reinterpret_cast<float*>(smem + Plan::off_sa);
  auto* sm_sum = reinterpret_cast<int32_t*>(smem + Plan::off_sum);
  auto* sm_sb = reinterpret_cast<half*>(smem + Plan::off_sb);
  auto* sm_zb = reinterpret_cast<half*>(smem + Plan::off_zb);
  TileScheduler visitor(sizes, problems, prefix);
  for (int tile = static_cast<int>(blockIdx.x);; tile += static_cast<int>(gridDim.x)) {
    const int p = visitor.get_problem_idx(tile);
    if (p < 0) break;
    const dim3 coord = visitor.get_tile_coord<TileCfg::BM, TileCfg::BN>(p, tile);
    const dim3 shape = sizes[p];
    const int M = static_cast<int>(shape.x), N = static_cast<int>(shape.y), K = static_cast<int>(shape.z);
    const int groups = K / kGroup;
    TileA a(ptr_a[p], M, K, packed_stride, coord.y, coord.x, wy, wz, tidx);
    TileB b(ptr_b[p], N, K, coord.y, coord.x, wx, wz, tidx);
    TileC c(ptr_c[p], M, N, output_ld, coord.y, coord.x, tidx);
    FragC accum{};
    FragA fa[TileCfg::WARP_ITER_K];
    FragB fb[TileCfg::WARP_ITER_K];
    float rsa[TileCfg::WARP_FRAG_M * 2];
    int32_t rsum[TileCfg::WARP_FRAG_M * 2];
    half rsb[TileCfg::WARP_FRAG_N * 2];
    half rzb[TileCfg::WARP_FRAG_N * 2];
    float out[TileCfg::WARP_FRAG_M * TileCfg::WARP_FRAG_N][TileCfg::FRAG_C_TH]{};
    for (int g = 0; g < groups; ++g) {
      a.load_smem(sm_a, true);
      b.load_smem(sm_b, true);
      commit_cp_async_group();
      if (tidx < kMixedBM) {
        const int r = static_cast<int>(coord.x) * kMixedBM + tidx;
        const bool valid = r < M;
        sm_sa[tidx] = valid ? ptr_as[p][r] : 0.0f;
        sm_sum[tidx] = valid ? ptr_sum[p][static_cast<int64_t>(r) * groups + g] : 0;
      }
      if (tidx < kMixedBN) {
        const int n = static_cast<int>(coord.y) * kMixedBN + tidx;
        const half* q = ptr_scale[p] + (static_cast<int64_t>(g) * N + n) * 2;
        sm_sb[tidx] = q[0];
        sm_zb[tidx] = q[1];
      }
      wait_cp_async_group<0>();
      __syncthreads();
#pragma unroll
      for (int k = 0; k < TileCfg::WARP_ITER_K; ++k) {
        a.load_frag(k, fa[k], sm_a);
        b.load_frag(k, fb[k], sm_b);
        TileCfg::mma_compute(accum, fa[k], fb[k]);
      }
      c.load_mixed_params(rsa, rsum, rsb, rzb, sm_sa, sm_sum, sm_sb, sm_zb, lane, wx, wy);
      c.template scale_and_correct<CENTER>(rsa, rsum, rsb, rzb, accum, out);
      c.clear_frag(accum);
      a.advance_gmem();
      b.advance_gmem();
      a.reset_frag_offset();
      b.reset_frag_offset();
      __syncthreads();
    }
    c.template write_frag_to_gmem<float>(out, lane, wx, wy, wz, smem);
    __syncthreads();
  }
}

#define DEFINE_LOWBIT(NAME, TILE, WBITS)                                                                          \
  extern "C" __global__ void NAME(half** a, half** b, half** s, half** c, float** as, int32_t** sum,             \
                                    dim3* sizes, int* prefix, int problems, int packed_stride, int output_ld) {     \
    lowbit_body<TILE, WBITS>(mxmoe_smem, a, b, s, c, as, sum, sizes, prefix, problems, packed_stride, output_ld); \
  }
DEFINE_LOWBIT(mxmoe_grouped_w2a4_g128_asym, TileA4, 2)
DEFINE_LOWBIT(mxmoe_grouped_w2a8_g128_asym, TileA8, 2)
DEFINE_LOWBIT(mxmoe_grouped_w4a4_g128_asym, TileA4, 4)
DEFINE_LOWBIT(mxmoe_grouped_w4a8_g128_asym, TileA8, 4)
DEFINE_LOWBIT(mxmoe_grouped_w8a8_g128_asym, TileA8, 8)
#undef DEFINE_LOWBIT

#define DEFINE_A16(NAME, TILE)                                                                                       \
  extern "C" __global__ void NAME(half** a, half** b, half** s, half** c, dim3* sizes, int* prefix, int problems,   \
                                  int output_ld) {                                                                  \
    mxmoe_grouped_body<TILE, true>(mxmoe_smem, a, b, s, c, sizes, prefix, problems, output_ld);                     \
  }
DEFINE_A16(mxmoe_grouped_mixed_w2a16_g128_asym, TileW2)
DEFINE_A16(mxmoe_grouped_mixed_w4a16_g128_asym, TileW4)
DEFINE_A16(mxmoe_grouped_mixed_w8a16_g128_asym, TileW8)
#undef DEFINE_A16

template <typename OffsetT>
__global__ void prepare_mixed_offsets(
    const OffsetT* offsets, const int* modes, int wanted, int E, int N, int K, int rows,
    const half* inputs, const int16_t* packed_input, int packed_stride,
    const int16_t* weights, int64_t weight_segment_words, const half* scale, int64_t scale_part_words,
    const float* activation_scale, const int32_t* activation_sum, half* output, int output_ld, int column_start,
    int part, int parts, int* prefix, dim3* sizes, half** pa, half** pb, half** ps, half** pc,
    float** pas, int32_t** psum) {
  for (int e = static_cast<int>(threadIdx.x); e < E; e += static_cast<int>(blockDim.x)) {
    const ExpertSpan span = expert_span(offsets, e, rows);
    const int m = modes[e] == wanted ? span.rows : 0;
    prefix[e] = cta_tiles(m, kMixedBM, N / kMixedBN);
    sizes[e] = dim3(m, N, K);
    pa[e] = wanted == 16 ? const_cast<half*>(inputs + span.start * K)
                         : reinterpret_cast<half*>(const_cast<int16_t*>(packed_input + span.start * packed_stride));
    pb[e] = reinterpret_cast<half*>(const_cast<int16_t*>(
        weights + (static_cast<int64_t>(e) * parts + part) * weight_segment_words));
    ps[e] = const_cast<half*>(scale + (static_cast<int64_t>(e) * parts + part) * scale_part_words);
    pc[e] = output + span.start * output_ld + column_start;
    pas[e] = const_cast<float*>(activation_scale + span.start);
    psum[e] = const_cast<int32_t*>(activation_sum + span.start * (K / kGroup));
  }
  inclusive_prefix(prefix, E);
}

template <int WBITS, int ABITS>
__global__ void repack_low(const int32_t* qweight, int16_t* packed, const int* modes,
                           int E, int parts, int Nfull, int Npart, int K, int64_t segment_words) {
  constexpr int source_pack = 32 / WBITS;
  constexpr int target_pack = 16 / ABITS;
  const int ep = static_cast<int>(blockIdx.y), e = ep / parts, part = ep % parts;
  if (e >= E || modes[part * E + e] != ABITS) return;
  const int64_t words = static_cast<int64_t>(Npart) * (K / target_pack);
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < words; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(index / (K / target_pack));
    const int word = static_cast<int>(index % (K / target_pack));
    uint16_t out = 0;
#pragma unroll
    for (int j = 0; j < target_pack; ++j) {
      const int k = word * target_pack + j;
      const int32_t raw = qweight[(static_cast<int64_t>(e) * Nfull + part * Npart + n) * (K / source_pack) + k / source_pack];
      const int q = unpack_code(static_cast<uint32_t>(raw), k % source_pack, WBITS) - (1 << (WBITS - 1));
      out = static_cast<uint16_t>((out << ABITS) | (static_cast<unsigned>(q) & ((1u << ABITS) - 1u)));
    }
    packed[(static_cast<int64_t>(e) * parts + part) * segment_words + index] =
        mxmoe_activation::bits_as_i16(out);
  }
}

template <int WBITS>
__global__ void repack_a16(const int32_t* qweight, int16_t* packed, const int* modes, int E, int parts, int Nfull,
                           int Npart, int K, int64_t segment_words) {
  constexpr int nblock = (16 / WBITS) * 8;
  const int ep = static_cast<int>(blockIdx.z);
  const int e = ep / parts;
  const int part = ep % parts;
  if (e >= E || modes[part * E + e] != 16) return;
  const int k0 = static_cast<int>(blockIdx.x) * 16;
  const int n0 = static_cast<int>(blockIdx.y) * nblock;
  __shared__ int orig[16 * nblock];
  __shared__ int perm_w[16 * nblock];
  __shared__ int iperm[32];
  const int64_t n_base = static_cast<int64_t>(e) * Nfull + part * Npart;
  const int64_t seg = (static_cast<int64_t>(e) * parts + part) * segment_words;
  repack_ct_a16_tile<WBITS>(qweight, n_base, K / (32 / WBITS), k0, n0, reinterpret_cast<uint16_t*>(packed + seg), K,
                            orig, perm_w, iperm);
}

template <int WBITS, typename ScaleT>
__global__ void pack_mixed_scale(const ScaleT* scales, const int32_t* qzeros, half* output, int E, int parts,
                                 int Nfull, int Npart, int G) {
  const int64_t total = static_cast<int64_t>(E) * parts * G * Npart;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t t = idx;
  const int n = static_cast<int>(t % Npart);
  t /= Npart;
  const int g = static_cast<int>(t % G);
  t /= G;
  const int part = static_cast<int>(t % parts);
  const int e = static_cast<int>(t / parts);
  write_group_scale<WBITS>(output + idx * 2, scales, qzeros, e, part * Npart + n, g, Nfull, G);
}

inline int mixed_sms_ready(int device) {
  const int sms = ensure_ready_device(device);
  static std::mutex mu; static std::array<std::atomic<int>, 32> ready{};
  if (ready[device].load(std::memory_order_acquire)) return sms;
  std::lock_guard<std::mutex> lock(mu);
  if (!ready[device].load(std::memory_order_relaxed)) {
#define SET_LOW(NAME, TILE) set_dyn_smem(reinterpret_cast<const void*>(NAME), LowPlan<TILE>::total, device)
    SET_LOW(mxmoe_grouped_w2a4_g128_asym, TileA4); SET_LOW(mxmoe_grouped_w4a4_g128_asym, TileA4);
    SET_LOW(mxmoe_grouped_w2a8_g128_asym, TileA8); SET_LOW(mxmoe_grouped_w4a8_g128_asym, TileA8);
    SET_LOW(mxmoe_grouped_w8a8_g128_asym, TileA8);
#undef SET_LOW
#define SET_A16(NAME, TILE) set_dyn_smem(reinterpret_cast<const void*>(NAME), SmemPlan<TILE>::total, device)
    SET_A16(mxmoe_grouped_mixed_w2a16_g128_asym, TileW2);
    SET_A16(mxmoe_grouped_mixed_w4a16_g128_asym, TileW4);
    SET_A16(mxmoe_grouped_mixed_w8a16_g128_asym, TileW8);
#undef SET_A16
    ready[device].store(1, std::memory_order_release);
  }
  return sms;
}

}  // namespace mxmoe_mixed

inline std::tuple<torch::Tensor, torch::Tensor> prepare_weights_mixed(
    torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros, int bits, torch::Tensor activation_bits) {
  using namespace mxmoe_mixed;
  bits = require_bits(bits);
  const NamedTensor weight_tensors[] = {{&qweight, "qweight"}, {&scales, "scales"}, {&qzeros, "qzeros"},
                                        {&activation_bits, "activation_bits"}};
  check_contiguous_then_device(qweight, weight_tensors);
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32 && qzeros.scalar_type() == torch::kInt32,
              "qweight/qzeros must be int32 CT storage");
  TORCH_CHECK(activation_bits.scalar_type() == torch::kInt32 && activation_bits.dim() == 2,
              "activation_bits must be int32 [parts,E]");
  TORCH_CHECK(qweight.dim() == 3 && scales.dim() == 3 && qzeros.dim() == 3, "invalid CT ranks");
  const int E = qweight.size(0), Nfull = qweight.size(1), parts = activation_bits.size(0);
  TORCH_CHECK(activation_bits.size(1) == E && (parts == 1 || parts == 2), "activation_bits must be [1|2,E]");
  TORCH_CHECK(Nfull % parts == 0, "weight N must divide into projection parts");
  const int Npart = Nfull / parts, p32 = 32 / bits, K = qweight.size(2) * p32, G = K / kGroup;
  check_aligned_nk(Npart, K, "prepare_weights_mixed");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({E, Nfull, G}), "scales shape mismatch");
  TORCH_CHECK(qzeros.sizes() == torch::IntArrayRef({E, Nfull / p32, G}), "qzeros shape mismatch");
  const int64_t segment_words = static_cast<int64_t>(Npart) * (K / 2);  // enough for expanded S8
  auto packed = torch::empty({E, parts, segment_words}, qweight.options().dtype(torch::kInt16));
  auto scale_zero = torch::empty({E, parts, G, Npart, 2}, qweight.options().dtype(torch::kFloat16));
  const int device = qweight.get_device(); mixed_sms_ready(device);
  c10::cuda::CUDAGuard guard(device); cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  const int ep = E * parts;
  TORCH_CHECK(static_cast<int64_t>(E) * parts <= 65535, "E*parts exceeds CUDA grid.y/z");
  if (E && ep) {
    const int32_t* q = qweight.data_ptr<int32_t>();
    int16_t* packed_out = packed.data_ptr<int16_t>();
    const int32_t* modes = activation_bits.data_ptr<int32_t>();
    const int64_t words4 = static_cast<int64_t>(Npart) * (K / 4);
    const int64_t words8 = static_cast<int64_t>(Npart) * (K / 2);
    const int blocks4 = std::min<int64_t>((words4 + 255) / 256, 65535);
    const int blocks8 = std::min<int64_t>((words8 + 255) / 256, 65535);
    dispatch_wbits(bits, [&](auto tag) {
      constexpr int W = decltype(tag)::value;
      repack_low<W, 4><<<dim3(blocks4, ep), 256, 0, stream>>>(q, packed_out, modes, E, parts, Nfull, Npart, K,
                                                              segment_words);
      repack_low<W, 8><<<dim3(blocks8, ep), 256, 0, stream>>>(q, packed_out, modes, E, parts, Nfull, Npart, K,
                                                              segment_words);
      repack_a16<W><<<dim3(K / 16, Npart / ((16 / W) * 8), ep), 128, 0, stream>>>(
          q, packed_out, modes, E, parts, Nfull, Npart, K, segment_words);
    });
    const int64_t total = static_cast<int64_t>(E) * parts * G * Npart;
    const int blocks = static_cast<int>((total + 255) / 256);
    half* dst = reinterpret_cast<half*>(scale_zero.data_ptr<at::Half>());
    const int32_t* zeros = qzeros.data_ptr<int32_t>();
    const bool launched = dispatch_scale_ptr(scales, [&](const auto* ptr) {
      dispatch_wbits(bits, [&](auto tag) {
        constexpr int W = decltype(tag)::value;
        pack_mixed_scale<W><<<blocks, 256, 0, stream>>>(ptr, zeros, dst, E, parts, Nfull, Npart, G);
      });
    });
    TORCH_CHECK(launched, "scales must be fp16/bf16/fp32");
    C10_CUDA_CHECK(cudaGetLastError());
  }
  return {packed, scale_zero};
}

inline void grouped_gemm_mixed_out(
    torch::Tensor inputs, torch::Tensor packed, torch::Tensor scale_zero, torch::Tensor offsets, int bits,
    torch::Tensor activation_bits, torch::Tensor packed_input, torch::Tensor activation_scales,
    torch::Tensor activation_sums, torch::Tensor output, torch::Tensor scratch,
    int column_start, int column_count, int activation_mask) {
  using namespace mxmoe_mixed;
  bits = require_bits(bits);
  for (auto pair : std::array<std::pair<const torch::Tensor*, const char*>, 10>{{
       {&inputs,"inputs"},{&packed,"packed"},{&scale_zero,"scale_zero"},{&offsets,"offsets"},
       {&activation_bits,"activation_bits"},{&packed_input,"packed_input"},{&activation_scales,"activation_scales"},
       {&activation_sums,"activation_sums"},{&output,"output"},{&scratch,"scratch"}}}) {
    check_cuda_contiguous(*pair.first, pair.second); check_same_device(*pair.first, inputs, pair.second);
  }
  TORCH_CHECK(inputs.scalar_type()==torch::kFloat16 && output.scalar_type()==torch::kFloat16, "inputs/output must be fp16");
  TORCH_CHECK(packed.scalar_type()==torch::kInt16 && scale_zero.scalar_type()==torch::kFloat16, "invalid packed weights");
  TORCH_CHECK(activation_bits.scalar_type()==torch::kInt32 && activation_bits.dim()==1, "activation_bits must be int32[E]");
  TORCH_CHECK(offsets.scalar_type()==torch::kInt || offsets.scalar_type()==torch::kLong, "offsets must be int32/int64");
  TORCH_CHECK(scratch.scalar_type()==torch::kUInt8, "scratch must come from workspace()");
  TORCH_CHECK(inputs.dim()==2 && output.dim()==2 && packed.dim()==3 && scale_zero.dim()==5, "invalid mixed tensor ranks");
  const int E=packed.size(0), parts=packed.size(1), K=inputs.size(1), rows=inputs.size(0), output_ld=output.size(1);
  const int Npart=scale_zero.size(3), G=K/kGroup, part=column_start/Npart;
  TORCH_CHECK(column_count==Npart && column_start==part*Npart && part>=0 && part<parts, "projection slice is not packed-aligned");
  TORCH_CHECK(scale_zero.sizes()==torch::IntArrayRef({E,parts,G,Npart,2}), "scale_zero mixed shape mismatch");
  TORCH_CHECK(packed.size(2)==static_cast<int64_t>(Npart)*(K/2), "packed weight segment mismatch");
  TORCH_CHECK(output.size(0)==rows && activation_bits.numel()==E && offsets.numel()==E+1, "routing/output shape mismatch");
  TORCH_CHECK(K>0 && K%kGroup==0 && Npart%kMixedBN==0, "mixed K/N must be multiples of 128");
  TORCH_CHECK(activation_mask>0 && (activation_mask&~7)==0, "activation mask must contain A4/A8/A16 bits");
  if (bits==8) TORCH_CHECK(!(activation_mask&1), "W8A4 is not supported; W8 requires A8 or A16");
  const bool low = activation_mask&3;
  if (low) {
    TORCH_CHECK(packed_input.scalar_type()==torch::kInt16 && packed_input.dim()==2 && packed_input.size(0)==rows,
                "low-bit packed_input must be int16[R,K/4|K/2]");
    TORCH_CHECK(packed_input.size(1)==K/4 || packed_input.size(1)==K/2, "packed_input stride mismatch");
    TORCH_CHECK(activation_scales.scalar_type()==torch::kFloat && activation_scales.numel()==rows,
                "activation_scales must be float32[R]");
    TORCH_CHECK(activation_sums.scalar_type()==torch::kInt32 &&
                activation_sums.sizes()==torch::IntArrayRef({rows,G}), "activation_sums must be int32[R,K/128]");
  }
  const auto layout=scratch_layout(E); TORCH_CHECK(scratch.numel()>=layout.bytes,"scratch is too small");
  const int device=inputs.get_device(), sms=mixed_sms_ready(device); if (!E || !rows) return;
  c10::cuda::CUDAGuard guard(device); cudaStream_t stream=c10::cuda::getCurrentCUDAStream(device);
  const ScratchView slot = scratch_view(scratch.data_ptr<uint8_t>(), layout);
  const int64_t seg = packed.size(2);
  const int64_t scale_part = static_cast<int64_t>(G) * Npart * 2;
  auto launch_prepare_mode = [&](int mode) {
    dispatch_offset_type(offsets, [&](auto tag) {
      using OffsetT = typename decltype(tag)::type;
      prepare_mixed_offsets<OffsetT><<<1, 256, 0, stream>>>(
          offsets.data_ptr<OffsetT>(), activation_bits.data_ptr<int32_t>(), mode, E, Npart, K, rows,
          reinterpret_cast<const half*>(inputs.data_ptr<at::Half>()),
          packed_input.numel() ? packed_input.data_ptr<int16_t>() : nullptr,
          packed_input.dim() == 2 ? static_cast<int>(packed_input.size(1)) : 0, packed.data_ptr<int16_t>(), seg,
          reinterpret_cast<const half*>(scale_zero.data_ptr<at::Half>()), scale_part,
          activation_scales.numel() ? activation_scales.data_ptr<float>() : nullptr,
          activation_sums.numel() ? activation_sums.data_ptr<int32_t>() : nullptr,
          reinterpret_cast<half*>(output.data_ptr<at::Half>()), output_ld, column_start, part, parts, slot.prefix,
          slot.sizes, slot.ptr_a, slot.ptr_b, slot.ptr_s, slot.ptr_c, slot.ptr_as, slot.ptr_sum);
    });
  };
#define LAUNCH_LOW(MODE, NAME, TILE)                                                                                 \
  do {                                                                                                               \
    launch_prepare_mode(MODE);                                                                                       \
    NAME<<<sms, dim3(32, LowPlan<TILE>::warps), LowPlan<TILE>::total, stream>>>(slot.ptr_a, slot.ptr_b, slot.ptr_s,  \
        slot.ptr_c, slot.ptr_as, slot.ptr_sum, slot.sizes, slot.prefix, E, packed_input.size(1), output_ld);         \
  } while (0)
  if (activation_mask & 1) {
    if (bits == 2) LAUNCH_LOW(4, mxmoe_grouped_w2a4_g128_asym, TileA4);
    else LAUNCH_LOW(4, mxmoe_grouped_w4a4_g128_asym, TileA4);
  }
  if (activation_mask & 2) {
    if (bits == 2) LAUNCH_LOW(8, mxmoe_grouped_w2a8_g128_asym, TileA8);
    else if (bits == 4) LAUNCH_LOW(8, mxmoe_grouped_w4a8_g128_asym, TileA8);
    else LAUNCH_LOW(8, mxmoe_grouped_w8a8_g128_asym, TileA8);
  }
#undef LAUNCH_LOW
#define LAUNCH_A16(NAME, TILE)                                                                                       \
  NAME<<<sms, dim3(32, SmemPlan<TILE>::warps), SmemPlan<TILE>::total, stream>>>(slot.ptr_a, slot.ptr_b, slot.ptr_s,  \
      slot.ptr_c, slot.sizes, slot.prefix, E, output_ld)
  if (activation_mask & 4) {
    launch_prepare_mode(16);
    if (bits == 2) LAUNCH_A16(mxmoe_grouped_mixed_w2a16_g128_asym, TileW2);
    else if (bits == 4) LAUNCH_A16(mxmoe_grouped_mixed_w4a16_g128_asym, TileW4);
    else LAUNCH_A16(mxmoe_grouped_mixed_w8a16_g128_asym, TileW8);
  }
#undef LAUNCH_A16
  C10_CUDA_CHECK(cudaGetLastError());
}
