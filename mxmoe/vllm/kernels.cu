// Serving binding for the existing MxMoE CTA. GEMM math is cta_gemm_wxa16g128 /
// QuantTileB / GemmTileA / GlobalTileC; this file only repacks, schedules, and launches.
//
// Tile (all bits): BM=64 BN=128 BK=128 WM=2 WN=2 WK=1 STAGE=3, MMA fp16->fp32.
// That W2 tile was previously exercised on A100 for M=1/17/65, N=K=256. W4/W8 reuse
// the same geometry so WARP_FRAG_N % PACK_NUM holds and group-128 scale advance
// (BK == GSIZE) matches cta_gemm_wxa16g128. WK=1 avoids the slice-k epilogue.
//
// Constraints: bits {2,4,8}, asymmetric group 128, FP16 A/C, K%128==0, N%128==0.
// M may be 0 or a non-multiple of 64. Empty experts are zero-tile problems.
// No cudaMalloc / host offset read / device sync on the grouped_gemm path.
#define MXMOE_DEVICE_ONLY

#include "cta_gemm.cuh"
#include "tile_scheduler.cuh"

#undef MAX
#undef OFFSET

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>


constexpr int kGroup = 128;
constexpr int kTileM = 64;
constexpr int kTileN = 128;

using TileW2 = tiled_gemm::TileConfig<
    kTileM, kTileN, 128, 2, 2, 1, 3, -1, tiled_gemm::MMA_FP16_FP32, tiled_gemm::NO_QUANT,
    tiled_gemm::QConfig<half, false, 2, kGroup, tiled_gemm::PackDim::MN, false, half, true>>;
using TileW4 = tiled_gemm::TileConfig<
    kTileM, kTileN, 128, 2, 2, 1, 3, -1, tiled_gemm::MMA_FP16_FP32, tiled_gemm::NO_QUANT,
    tiled_gemm::QConfig<half, false, 4, kGroup, tiled_gemm::PackDim::MN, false, half, true>>;
using TileW8 = tiled_gemm::TileConfig<
    kTileM, kTileN, 128, 2, 2, 1, 3, -1, tiled_gemm::MMA_FP16_FP32, tiled_gemm::NO_QUANT,
    tiled_gemm::QConfig<half, false, 8, kGroup, tiled_gemm::PackDim::MN, false, half, true>>;

template <typename TileCfg>
struct SmemPlan {
  using TileA = tiled_gemm::GemmTileA<half, tiled_gemm::Layout::RowMajor, TileCfg>;
  using TileB = tiled_gemm::QuantTileB<half, tiled_gemm::Layout::ColMajor, TileCfg, typename TileCfg::QConfigB>;
  using TileC = tiled_gemm::GlobalTileC<half, tiled_gemm::Layout::RowMajor, TileCfg>;
  static constexpr int tile_bytes =
      TileCfg::STAGE * (TileA::VEC_PER_ROW * TileA::TILE_ROW + TileB::VEC_PER_ROW * TileB::TILE_ROW) * 16;
  static constexpr int scale_bytes = TileCfg::STAGE * TileB::G2S_SCALE_SIZE * static_cast<int>(sizeof(half));
  static constexpr int total = tile_bytes + scale_bytes;
  static constexpr int warps = TileCfg::WARP_M * TileCfg::WARP_N * TileCfg::WARP_K;
};

static_assert(SmemPlan<TileW2>::total > 0 && SmemPlan<TileW2>::total < 163840, "w2 smem");
static_assert(SmemPlan<TileW4>::total > 0 && SmemPlan<TileW4>::total < 163840, "w4 smem");
static_assert(SmemPlan<TileW8>::total > 0 && SmemPlan<TileW8>::total < 163840, "w8 smem");
static_assert(TileW2::QConfigB::RAW_ZERO && TileW2::QConfigB::QBITS == 2, "w2 raw zero");
static_assert(TileW4::QConfigB::QBITS == 4 && TileW8::QConfigB::QBITS == 8, "bit tags");
static_assert(TileW2::BK == kGroup && TileW2::WARP_K == 1, "group tile");

extern __shared__ __align__(16) uint8_t mxmoe_smem[];

template <typename TileCfg, bool Sliced = false>
__device__ __forceinline__ void mxmoe_grouped_body(uint8_t* smem, half** ptr_As, half** ptr_Bs, half** ptr_scale,
                                                    half** ptr_Cs, dim3* problem_sizes, int* prefix, int problem_count,
                                                    int output_ld) {
  using Plan = SmemPlan<TileCfg>;
  using TileA = typename Plan::TileA;
  using TileB = typename Plan::TileB;
  using TileC = typename Plan::TileC;
  static_assert(TileB::GSIZE == kGroup, "group 128");
  static_assert(TileCfg::BK == kGroup, "scale advances one group per BK");
  if (problem_count <= 0) return;

  const int lane = threadIdx.x;
  const int tidx = lane + threadIdx.y * WarpSize;
  half* smem_scale = reinterpret_cast<half*>(smem + Plan::tile_bytes);
  const auto warp_x = TileCfg::warp_x();
  const auto warp_y = TileCfg::warp_y();
  const auto warp_z = TileCfg::warp_z();
  TileScheduler visitor(problem_sizes, problem_count, prefix);
  for (int tile_idx = static_cast<int>(blockIdx.x);; tile_idx += static_cast<int>(gridDim.x)) {
    const int problem_idx = visitor.get_problem_idx(tile_idx);
    if (problem_idx < 0) break;
    // x = M tile, y = N tile. Match compose_kernel.py's bx/by argument order.
    const dim3 tile_coord = visitor.get_tile_coord<TileCfg::BM, TileCfg::BN>(problem_idx, tile_idx);
    const dim3 problem = problem_sizes[problem_idx];
    const int M = static_cast<int>(problem.x);
    const int N = static_cast<int>(problem.y);
    const int K = static_cast<int>(problem.z);
    TileA tile_a(ptr_As[problem_idx], M, K, tile_coord.y, tile_coord.x, warp_y, warp_z, tidx);
    TileB tile_b(ptr_Bs[problem_idx], N, K, tile_coord.y, tile_coord.x, warp_x, warp_z, tidx);
    TileC tile_c(ptr_Cs[problem_idx], M, N, tile_coord.y, tile_coord.x, tidx);
    if constexpr (Sliced) {
      tile_c.leading = output_ld;
      tile_c.cta_C = ptr_Cs[problem_idx] +
          static_cast<int64_t>(tile_coord.x) * TileCfg::BM * output_ld + tile_coord.y * TileCfg::BN;
    }
    const half* gmem_scale = ptr_scale[problem_idx] + tile_coord.y * TileB::CTA_SCALE_SIZE;
    tiled_gemm::cta_gemm_wxa16g128<TileCfg>(tile_a, tile_b, tile_c, smem, gmem_scale, smem_scale, N, K, lane, warp_x,
                                            warp_y, warp_z, tidx);
    // Persistent CTAs may immediately reuse the same shared-memory stages.
    __syncthreads();
  }
}

#define DEFINE_GROUPED(NAME, TILE)                                                                                   \
  extern "C" __global__ void NAME(half** ptr_As, half** ptr_Bs, half** ptr_scale, half** ptr_Cs, dim3* problem_sizes, \
                                  int* prefix, int problem_count) {                                                 \
    mxmoe_grouped_body<TILE>(mxmoe_smem, ptr_As, ptr_Bs, ptr_scale, ptr_Cs, problem_sizes, prefix, problem_count, 0); \
  }
DEFINE_GROUPED(mxmoe_grouped_w2a16_g128_asym, TileW2)
DEFINE_GROUPED(mxmoe_grouped_w4a16_g128_asym, TileW4)
DEFINE_GROUPED(mxmoe_grouped_w8a16_g128_asym, TileW8)
#undef DEFINE_GROUPED

namespace {

struct ExpertSpan {
  int64_t start;
  int rows;
};

template <typename OffsetT>
__device__ __forceinline__ ExpertSpan expert_span(const OffsetT* offsets, int expert, int rows) {
  const int64_t start = static_cast<int64_t>(offsets[expert]);
  const int64_t end = static_cast<int64_t>(offsets[expert + 1]);
  assert(start >= 0 && end >= start && end <= rows);
  return {start, static_cast<int>(end - start)};
}

__device__ __forceinline__ int cta_tiles(int rows, int tile_m, int tiles_n) {
  return ((rows + tile_m - 1) / tile_m) * tiles_n;
}

__device__ __forceinline__ void inclusive_prefix(int* prefix, int count) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int i = 0; i < count; ++i) {
      acc += prefix[i];
      prefix[i] = acc;
    }
  }
}

template <typename OffsetT>
__global__ void mxmoe_prepare_grouped_offsets(const OffsetT* __restrict__ offsets, int E, int N, int K, int rows,
                                              int tile_m, int tiles_n, const half* __restrict__ inputs,
                                              const half* __restrict__ packed, const half* __restrict__ scale,
                                              half* __restrict__ output, int64_t packed_stride, int64_t scale_stride,
                                              int* __restrict__ prefix, dim3* __restrict__ problem_sizes,
                                              half** __restrict__ ptr_a, half** __restrict__ ptr_b,
                                              half** __restrict__ ptr_s, half** __restrict__ ptr_c) {
  for (int e = static_cast<int>(threadIdx.x); e < E; e += static_cast<int>(blockDim.x)) {
    const ExpertSpan span = expert_span(offsets, e, rows);
    prefix[e] = cta_tiles(span.rows, tile_m, tiles_n);
    problem_sizes[e] = dim3(span.rows, N, K);
    ptr_a[e] = const_cast<half*>(inputs + span.start * K);
    ptr_b[e] = const_cast<half*>(packed + static_cast<int64_t>(e) * packed_stride);
    ptr_s[e] = const_cast<half*>(scale + static_cast<int64_t>(e) * scale_stride);
    ptr_c[e] = output + span.start * N;
  }
  inclusive_prefix(prefix, E);
}

__device__ __forceinline__ int unpack_code(uint32_t word, int sub, int bits) {
  return static_cast<int>((word >> (sub * bits)) & ((1u << bits) - 1u));
}

// Transcription of quantize.cuh permute_weight (Row) + pack_weightonly. jj is always 0
// because FRAG_N == 8. High bits of the uint16 are frag_id 0.
template <int BITS>
__device__ void permute_pack_tile(const int* orig, int* perm_w, int* iperm) {
  constexpr int kPack = 16 / BITS;
  if constexpr (BITS == 8) {
    constexpr int proj[4] = {1, 0, 3, 2};
    constexpr int desired[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    compose_perm_indices<8, 4>(proj, desired, iperm);
  } else if constexpr (BITS == 4) {
    constexpr int proj[8] = {3, 7, 2, 6, 1, 5, 0, 4};
    constexpr int desired[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    compose_perm_indices<16, 8>(proj, desired, iperm);
  } else {
    constexpr int proj[16] = {7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8};
    constexpr int desired[32] = {0, 8,  16, 24, 1, 9,  17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
                                 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};
    compose_perm_indices<32, 16>(proj, desired, iperm);
  }

  for (int trav_j = 0; trav_j < 8; ++trav_j) {
    for (int trav_i = 0; trav_i < 8; trav_i += 2) {
      int buff_val[32];
      int buff_loc[32];
      int count = 0;
      for (int ii = 0; ii < 16; ii += 8) {
        for (int trav_ii = 0; trav_ii < 2; ++trav_ii) {
          for (int frag_id = 0; frag_id < kPack; ++frag_id) {
            const int k_local = ii + trav_i + trav_ii;
            const int n_local = frag_id * 8 + trav_j;
            const int loc = k_local + n_local * 16;
            buff_val[count] = orig[loc];
            buff_loc[count] = loc;
            ++count;
          }
        }
      }
      for (int x = 0; x < count; ++x) {
        perm_w[buff_loc[x]] = buff_val[iperm[x]];
      }
    }
  }
}

template <int BITS>
__device__ __forceinline__ void repack_ct_a16_tile(const int32_t* qweight, int64_t n_base, int k_pack, int k0, int n0,
                                                   uint16_t* out_rows, int K, int* orig, int* perm_w, int* iperm) {
  constexpr int kPack16 = 16 / BITS;
  constexpr int kPack32 = 32 / BITS;
  constexpr int kMask = (1 << BITS) - 1;
  constexpr int kNBlock = kPack16 * 8;
  for (int idx = static_cast<int>(threadIdx.x); idx < 16 * kNBlock; idx += static_cast<int>(blockDim.x)) {
    const int k_local = idx % 16;
    const int n_local = idx / 16;
    const int k = k0 + k_local;
    const int n = n0 + n_local;
    const int32_t word = qweight[(n_base + n) * k_pack + (k / kPack32)];
    orig[idx] = unpack_code(static_cast<uint32_t>(word), k % kPack32, BITS);
    perm_w[idx] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) permute_pack_tile<BITS>(orig, perm_w, iperm);
  __syncthreads();
  for (int idx = static_cast<int>(threadIdx.x); idx < 16 * 8; idx += static_cast<int>(blockDim.x)) {
    const int k_local = idx % 16;
    const int jj = idx / 16;
    uint16_t pack = 0;
    for (int frag_id = 0; frag_id < kPack16; ++frag_id) {
      pack = static_cast<uint16_t>((pack << BITS) | (perm_w[k_local + (frag_id * 8 + jj) * 16] & kMask));
    }
    out_rows[static_cast<int64_t>(n0 / kPack16 + jj) * K + (k0 + k_local)] = pack;
  }
}

template <int BITS>
__global__ void mxmoe_repack_ct_wna16(const int32_t* __restrict__ qweight, uint16_t* __restrict__ packed, int N, int K) {
  constexpr int kNBlock = (16 / BITS) * 8;
  const int e = static_cast<int>(blockIdx.z);
  const int k0 = static_cast<int>(blockIdx.x) * 16;
  const int n0 = static_cast<int>(blockIdx.y) * kNBlock;
  __shared__ int orig[16 * kNBlock];
  __shared__ int perm_w[16 * kNBlock];
  __shared__ int iperm[32];
  repack_ct_a16_tile<BITS>(qweight, static_cast<int64_t>(e) * N, K / (32 / BITS), k0, n0,
                           packed + static_cast<int64_t>(e) * (N / (16 / BITS)) * K, K, orig, perm_w, iperm);
}

template <typename ScaleT>
__device__ __forceinline__ half scale_to_half(ScaleT value) {
  if constexpr (std::is_same_v<ScaleT, float>) {
    return __float2half_rn(value);
  } else if constexpr (std::is_same_v<ScaleT, half>) {
    return value;
  } else {
    return __float2half_rn(__bfloat162float(value));
  }
}

template <int BITS, typename ScaleT>
__device__ __forceinline__ void write_group_scale(half* dst, const ScaleT* scales, const int32_t* qzeros, int e, int n,
                                                  int g, int N, int G) {
  constexpr int kPack32 = 32 / BITS;
  const int zp_n = n / kPack32;
  const int32_t zword = qzeros[(static_cast<int64_t>(e) * (N / kPack32) + zp_n) * G + g];
  dst[0] = scale_to_half(scales[(static_cast<int64_t>(e) * N + n) * G + g]);
  dst[1] = __int2half_rn(unpack_code(static_cast<uint32_t>(zword), n % kPack32, BITS));
}

template <int BITS, typename ScaleT>
__global__ void mxmoe_pack_scale_rawzero(const ScaleT* __restrict__ scales, const int32_t* __restrict__ qzeros,
                                         half* __restrict__ scale_zero, int E, int N, int G) {
  const int64_t total = static_cast<int64_t>(E) * N * G;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int n = static_cast<int>(idx % N);
  const int g = static_cast<int>((idx / N) % G);
  const int e = static_cast<int>(idx / (static_cast<int64_t>(N) * G));
  write_group_scale<BITS>(scale_zero + ((static_cast<int64_t>(e) * G + g) * N + n) * 2, scales, qzeros, e, n, g, N, G);
}

struct ScratchLayout {
  int64_t prefix = 0;
  int64_t sizes = 0;
  int64_t ptr_a = 0;
  int64_t ptr_b = 0;
  int64_t ptr_s = 0;
  int64_t ptr_c = 0;
  int64_t ptr_as = 0;
  int64_t ptr_sum = 0;
  int64_t bytes = 0;
};

ScratchLayout scratch_layout(int64_t experts) {
  auto align16 = [](int64_t x) { return (x + 15) & ~int64_t{15}; };
  ScratchLayout layout;
  int64_t off = 0;
  const int64_t n = experts > 0 ? experts : 1;
  auto push = [&](int64_t bytes) {
    const int64_t at = off;
    off = align16(off + bytes);
    return at;
  };
  layout.prefix = push(static_cast<int64_t>(sizeof(int)) * n);
  layout.sizes = push(static_cast<int64_t>(sizeof(dim3)) * n);
  layout.ptr_a = push(static_cast<int64_t>(sizeof(half*)) * n);
  layout.ptr_b = push(static_cast<int64_t>(sizeof(half*)) * n);
  layout.ptr_s = push(static_cast<int64_t>(sizeof(half*)) * n);
  layout.ptr_c = push(static_cast<int64_t>(sizeof(half*)) * n);
  layout.ptr_as = push(static_cast<int64_t>(sizeof(float*)) * n);
  layout.ptr_sum = push(static_cast<int64_t>(sizeof(int32_t*)) * n);
  layout.bytes = off;
  return layout;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& tensor, const torch::Tensor& ref, const char* name) {
  TORCH_CHECK(tensor.device() == ref.device(), name, " is on ", tensor.device(), ", expected ", ref.device());
}

struct NamedTensor { const torch::Tensor* tensor; const char* name; };

template <size_t N>
void check_device_except(const torch::Tensor& ref, const NamedTensor (&tensors)[N]) {
  for (const NamedTensor& item : tensors)
    if (item.tensor != &ref) check_same_device(*item.tensor, ref, item.name);
}

template <bool RequireDefined = false, size_t N>
void check_contiguous_then_device(const torch::Tensor& ref, const NamedTensor (&tensors)[N]) {
  for (const NamedTensor& item : tensors) {
    if constexpr (RequireDefined) {
      TORCH_CHECK(item.tensor->defined(), item.name, " is undefined");
    }
    check_cuda_contiguous(*item.tensor, item.name);
  }
  check_device_except(ref, tensors);
}

template <typename Fn>
void dispatch_wbits(int bits, Fn&& fn) {
  if (bits == 2) fn(std::integral_constant<int, 2>{});
  else if (bits == 4) fn(std::integral_constant<int, 4>{});
  else fn(std::integral_constant<int, 8>{});
}

template <typename T> struct TypeTag { using type = T; };

template <typename Fn>
void dispatch_offset_type(const torch::Tensor& offsets, Fn&& fn) {
  if (offsets.scalar_type() == torch::kLong) fn(TypeTag<int64_t>{});
  else fn(TypeTag<int32_t>{});
}

template <typename Fn>
bool dispatch_scale_ptr(const torch::Tensor& scales, Fn&& fn) {
  if (scales.scalar_type() == torch::kFloat) fn(scales.data_ptr<float>());
  else if (scales.scalar_type() == torch::kFloat16) fn(reinterpret_cast<const half*>(scales.data_ptr<at::Half>()));
  else if (scales.scalar_type() == torch::kBFloat16)
    fn(reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()));
  else return false;
  return true;
}

struct ScratchView {
  int* prefix;
  dim3* sizes;
  half** ptr_a;
  half** ptr_b;
  half** ptr_s;
  half** ptr_c;
  float** ptr_as;
  int32_t** ptr_sum;
};

ScratchView scratch_view(uint8_t* base, const ScratchLayout& layout) {
  return {reinterpret_cast<int*>(base + layout.prefix), reinterpret_cast<dim3*>(base + layout.sizes),
          reinterpret_cast<half**>(base + layout.ptr_a), reinterpret_cast<half**>(base + layout.ptr_b),
          reinterpret_cast<half**>(base + layout.ptr_s), reinterpret_cast<half**>(base + layout.ptr_c),
          reinterpret_cast<float**>(base + layout.ptr_as), reinterpret_cast<int32_t**>(base + layout.ptr_sum)};
}

void launch_uniform_a16(int bits, int sms, half** a, half** b, half** s, half** c, dim3* sizes, int* prefix,
                        int problems, cudaStream_t stream) {
#define LAUNCH_UNIFORM(KERNEL, TILE)                                                                             \
  KERNEL<<<sms, dim3(32, SmemPlan<TILE>::warps), SmemPlan<TILE>::total, stream>>>(a, b, s, c, sizes, prefix, problems)
  if (bits == 2) LAUNCH_UNIFORM(mxmoe_grouped_w2a16_g128_asym, TileW2);
  else if (bits == 4) LAUNCH_UNIFORM(mxmoe_grouped_w4a16_g128_asym, TileW4);
  else LAUNCH_UNIFORM(mxmoe_grouped_w8a16_g128_asym, TileW8);
#undef LAUNCH_UNIFORM
}

int require_bits(int bits) {
  TORCH_CHECK(bits == 2 || bits == 4 || bits == 8, "mxmoe supports uniform asymmetric W2/W4/W8 only, got bits=", bits);
  return bits;
}

void check_aligned_nk(int64_t N, int64_t K, const char* what) {
  TORCH_CHECK(K > 0 && K % kGroup == 0, what, " K=", K, " must be a positive multiple of ", kGroup);
  TORCH_CHECK(N > 0 && N % kTileN == 0, what,
              " N=", N, " must be a positive multiple of ", kTileN,
              " (MxMoE tile BN / pack fragment). Pad this projection before prepare_weights; "
              "the op does not pad.");
  TORCH_CHECK(N <= std::numeric_limits<int>::max() && K <= std::numeric_limits<int>::max(), what, " N/K exceed int32");
}

// GemmTileA/QuantTileB/GlobalTileC form tile bases and in-tile addresses in signed int.
// Expert-start pointer math is int64 and does not protect those offsets.
void check_int_product(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0, what, " span is negative");
  TORCH_CHECK(b == 0 || a <= static_cast<int64_t>(std::numeric_limits<int>::max()) / b,
              what, " element span ", a, "*", b,
              " exceeds signed int. Reused MxMoE tiles index A/B/C in int; "
              "limit is max(M, 64)*K, max(M, 64)*N, and (N/(16/bits))*K.");
}

void set_dyn_smem(const void* fn, int bytes, int device) {
  int max_smem = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(bytes <= max_smem, "MxMoE dynamic smem ", bytes, " exceeds device ", device, " opt-in max ", max_smem);
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

int ensure_ready_device(int device) {
  TORCH_CHECK(device >= 0 && device < 32, "mxmoe device index must be in [0, 32)");
  static std::mutex mu;
  static std::array<std::atomic<int>, 32> sm_counts{};
  int sms = sm_counts[static_cast<size_t>(device)].load(std::memory_order_acquire);
  if (sms != 0) return sms;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone,
              "mxmoe kernel attributes must be set before CUDA graph capture; call prepare_weights or workspace first");

  std::lock_guard<std::mutex> lock(mu);
  sms = sm_counts[static_cast<size_t>(device)].load(std::memory_order_relaxed);
  if (sms != 0) return sms;
  c10::cuda::CUDAGuard guard(device);
  set_dyn_smem(reinterpret_cast<const void*>(mxmoe_grouped_w2a16_g128_asym), SmemPlan<TileW2>::total, device);
  set_dyn_smem(reinterpret_cast<const void*>(mxmoe_grouped_w4a16_g128_asym), SmemPlan<TileW4>::total, device);
  set_dyn_smem(reinterpret_cast<const void*>(mxmoe_grouped_w8a16_g128_asym), SmemPlan<TileW8>::total, device);
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(sms > 0, "CUDA device reported no SMs");
  sm_counts[static_cast<size_t>(device)].store(sms, std::memory_order_release);
  return sms;
}

template <int BITS>
void launch_repack(const torch::Tensor& qweight, torch::Tensor& packed, int E, int N, int K, cudaStream_t stream) {
  constexpr int kPack16 = 16 / BITS;
  constexpr int kNBlock = kPack16 * 8;
  const int k_tiles = K / 16;
  const int n_tiles = N / kNBlock;
  if (E == 0 || k_tiles == 0 || n_tiles == 0) return;
  TORCH_CHECK(k_tiles <= 2147483647 && n_tiles <= 65535 && E <= 65535, "CT repack grid exceeds CUDA limits");
  mxmoe_repack_ct_wna16<BITS><<<dim3(k_tiles, n_tiles, E), 128, 0, stream>>>(
      qweight.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(packed.data_ptr<int16_t>()), N, K);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int BITS, typename ScaleT>
void launch_scale_kernel(const ScaleT* scales, const int32_t* qzeros, half* scale_zero, int E, int N, int G,
                         cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(E) * N * G;
  if (total == 0) return;
  constexpr int kThreads = 256;
  TORCH_CHECK(total <= static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreads, "scale pack grid too large");
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  mxmoe_pack_scale_rawzero<BITS, ScaleT><<<blocks, kThreads, 0, stream>>>(scales, qzeros, scale_zero, E, N, G);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int BITS>
void launch_scale(const torch::Tensor& scales, const torch::Tensor& qzeros, torch::Tensor& scale_zero, int E, int N,
                  int K, cudaStream_t stream) {
  const int G = K / kGroup;
  half* dst = reinterpret_cast<half*>(scale_zero.data_ptr<at::Half>());
  const int32_t* zeros = qzeros.data_ptr<int32_t>();
  const bool launched = dispatch_scale_ptr(scales, [&](const auto* ptr) {
    launch_scale_kernel<BITS>(ptr, zeros, dst, E, N, G, stream);
  });
  TORCH_CHECK(launched, "scales must be fp16, bf16, or fp32, got ", scales.scalar_type());
}


template <typename OffsetT>
void launch_prepare(const torch::Tensor& offsets, int E, int N, int K, int rows, const half* inputs, const half* packed,
                    const half* scale, half* output, int64_t packed_stride, int64_t scale_stride, int* prefix,
                    dim3* sizes, half** ptr_a, half** ptr_b, half** ptr_s, half** ptr_c, cudaStream_t stream) {
  mxmoe_prepare_grouped_offsets<OffsetT><<<1, 256, 0, stream>>>(
      offsets.data_ptr<OffsetT>(), E, N, K, rows, kTileM, N / kTileN, inputs, packed, scale, output, packed_stride,
      scale_stride, prefix, sizes, ptr_a, ptr_b, ptr_s, ptr_c);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

#include "activation.cuh"
#include "mixed_gemm.cuh"

int64_t workspace_bytes(int64_t experts) {
  TORCH_CHECK(experts >= 0, "experts must be >= 0");
  TORCH_CHECK(experts <= (1 << 20), "experts exceeds mxmoe workspace limit");
  return scratch_layout(experts).bytes;
}

void ensure_ready(int device) { ensure_ready_device(device); }

std::tuple<torch::Tensor, torch::Tensor> prepare_weights(torch::Tensor qweight, torch::Tensor scales,
                                                         torch::Tensor qzeros, int bits) {
  bits = require_bits(bits);
  const NamedTensor weight_tensors[] = {{&qweight, "qweight"}, {&scales, "scales"}, {&qzeros, "qzeros"}};
  check_contiguous_then_device(qweight, weight_tensors);
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32 CT packed codes");
  TORCH_CHECK(qzeros.scalar_type() == torch::kInt32, "qzeros must be int32 CT packed raw zeros");
  TORCH_CHECK(qweight.dim() == 3 && scales.dim() == 3 && qzeros.dim() == 3, "qweight, scales, qzeros must be rank 3");

  const int pack32 = 32 / bits;
  const int E = static_cast<int>(qweight.size(0));
  const int N = static_cast<int>(qweight.size(1));
  TORCH_CHECK(qweight.size(2) > 0 && qweight.size(2) <= std::numeric_limits<int>::max() / pack32,
              "qweight packed-K dimension is invalid");
  const int K = static_cast<int>(qweight.size(2) * pack32);
  check_aligned_nk(N, K, "prepare_weights");
  check_int_product(static_cast<int64_t>(N) / (16 / bits), K, "prepare_weights packed N/(16/bits)*K");
  TORCH_CHECK(E >= 0 && E <= 65535, "E must be in [0, 65535]");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({E, N, K / kGroup}),
              "scales must be [E, N, K/128]=[", E, ", ", N, ", ", K / kGroup, "], got ", scales.sizes());
  TORCH_CHECK(qzeros.sizes() == torch::IntArrayRef({E, N / pack32, K / kGroup}),
              "qzeros must be [E, N/(32/bits), K/128]=[", E, ", ", N / pack32, ", ", K / kGroup, "], got ",
              qzeros.sizes());

  const auto device = qweight.device();
  ensure_ready_device(device.index());
  auto packed = torch::empty({E, N / (16 / bits), K}, qweight.options().dtype(torch::kInt16));
  auto scale_zero = torch::empty({E, K / kGroup, N, 2}, qweight.options().dtype(torch::kFloat16));
  if (E == 0) return {packed, scale_zero};

  c10::cuda::CUDAGuard guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());
  dispatch_wbits(bits, [&](auto tag) {
    constexpr int kBits = decltype(tag)::value;
    launch_repack<kBits>(qweight, packed, E, N, K, stream);
    launch_scale<kBits>(scales, qzeros, scale_zero, E, N, K, stream);
  });
  return {packed, scale_zero};
}

void grouped_gemm_out(torch::Tensor inputs, torch::Tensor packed, torch::Tensor scale_zero, torch::Tensor offsets,
                      int bits, torch::Tensor output, torch::Tensor scratch) {
  bits = require_bits(bits);
  const NamedTensor gemm_tensors[] = {{&inputs, "inputs"}, {&packed, "packed"}, {&scale_zero, "scale_zero"},
                                      {&offsets, "offsets"}, {&output, "output"}, {&scratch, "scratch"}};
  check_contiguous_then_device(inputs, gemm_tensors);

  TORCH_CHECK(inputs.scalar_type() == torch::kFloat16, "inputs must be fp16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be fp16");
  TORCH_CHECK(scale_zero.scalar_type() == torch::kFloat16, "scale_zero must be fp16");
  TORCH_CHECK(packed.scalar_type() == torch::kInt16 || packed.scalar_type() == torch::kUInt16,
              "packed must be int16/uint16 bit storage, not ", packed.scalar_type());
  TORCH_CHECK(scratch.scalar_type() == torch::kUInt8, "scratch must be the uint8 tensor returned by workspace()");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong || offsets.scalar_type() == torch::kInt,
              "offsets must be int32 or int64");
  TORCH_CHECK(inputs.dim() == 2 && output.dim() == 2, "inputs and output must be rank 2");
  TORCH_CHECK(packed.dim() == 3 && scale_zero.dim() == 4, "packed must be rank 3 and scale_zero rank 4");

  const int E = static_cast<int>(packed.size(0));
  const int64_t packed_n64 = packed.size(1);
  const int64_t k64 = packed.size(2);
  const int pack16 = 16 / bits;
  TORCH_CHECK(k64 > 0 && k64 <= std::numeric_limits<int>::max(), "packed K exceeds int32");
  TORCH_CHECK(packed_n64 >= 0 && packed_n64 <= std::numeric_limits<int>::max() / pack16, "packed N exceeds int32");
  const int packed_n = static_cast<int>(packed_n64);
  const int K = static_cast<int>(k64);
  const int N = packed_n * pack16;
  check_aligned_nk(N, K, "grouped_gemm_out");
  check_int_product(packed_n64, k64, "grouped_gemm_out packed N/(16/bits)*K");
  TORCH_CHECK(inputs.size(1) == K, "inputs K ", inputs.size(1), " != packed K ", K);
  TORCH_CHECK(output.size(0) == inputs.size(0) && output.size(1) == N,
              "output must be [routed_rows, N]=[", inputs.size(0), ", ", N, "], got ", output.sizes());
  TORCH_CHECK(inputs.size(0) <= std::numeric_limits<int>::max(), "routed_rows exceeds int32");
  const int64_t span_m = std::max(inputs.size(0), static_cast<int64_t>(kTileM));
  check_int_product(span_m, K, "grouped_gemm_out activation M*K");
  check_int_product(span_m, N, "grouped_gemm_out output M*N");
  TORCH_CHECK(scale_zero.sizes() == torch::IntArrayRef({E, K / kGroup, N, 2}),
              "scale_zero must be [E, K/128, N, 2]=[", E, ", ", K / kGroup, ", ", N, ", 2], got ", scale_zero.sizes());
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == E + 1, "offsets must be int GPU [E+1], got ", offsets.sizes(),
              " for E=", E);
  TORCH_CHECK(E >= 0 && E <= 65535, "E must be in [0, 65535]");
  const auto layout = scratch_layout(E);
  TORCH_CHECK(scratch.numel() >= layout.bytes, "scratch has ", scratch.numel(), " bytes, need ", layout.bytes,
              " from workspace(E=", E, "). One scratch is one in-flight slot.");

  const int device = inputs.get_device();
  const int sms = ensure_ready_device(device);
  if (E == 0 || inputs.size(0) == 0) return;

  c10::cuda::CUDAGuard guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);

  const ScratchView slot = scratch_view(scratch.data_ptr<uint8_t>(), layout);
  const half* inputs_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  const half* packed_ptr = reinterpret_cast<const half*>(packed.data_ptr());
  const half* scale_ptr = reinterpret_cast<const half*>(scale_zero.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  const int64_t packed_stride = static_cast<int64_t>(packed_n) * K;
  const int64_t scale_stride = static_cast<int64_t>(K / kGroup) * N * 2;
  const int rows = static_cast<int>(inputs.size(0));
  dispatch_offset_type(offsets, [&](auto tag) {
    using OffsetT = typename decltype(tag)::type;
    launch_prepare<OffsetT>(offsets, E, N, K, rows, inputs_ptr, packed_ptr, scale_ptr, output_ptr, packed_stride,
                            scale_stride, slot.prefix, slot.sizes, slot.ptr_a, slot.ptr_b, slot.ptr_s, slot.ptr_c,
                            stream);
  });
  launch_uniform_a16(bits, sms, slot.ptr_a, slot.ptr_b, slot.ptr_s, slot.ptr_c, slot.sizes, slot.prefix, E, stream);
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare_weights", &prepare_weights, "Repack CT WNA16 codes to MxMoE packed layout and raw-zero scale");
  m.def("workspace_bytes", &workspace_bytes, "Byte size of one routing-independent scratch slot");
  m.def("ensure_ready", &ensure_ready, "Set per-device dynamic smem attributes before graph capture");
  m.def("prepare_weights_mixed", &prepare_weights_mixed,
        "Repack exact CT codes per projection for MxMoE A4/A8/A16");
  m.def("quantize_activations_out", &quantize_activations_out,
        "Dynamic per-token MxMoE activation preparation");
  m.def("grouped_gemm_mixed_out", &grouped_gemm_mixed_out,
        "Current-stream mixed-activation MxMoE grouped GEMM");
  m.def("grouped_gemm_out", &grouped_gemm_out, "Current-stream MxMoE grouped GEMM into the caller output buffer");
}
