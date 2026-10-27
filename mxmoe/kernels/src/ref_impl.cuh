#pragma once

#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/integer_subbyte.h>
#include "cta_gemm.cuh"
#include "layout.cuh"

using cutlass::gemm::GemmCoord;
using cutlass::gemm::kernel::GroupScheduleMode;

template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

namespace mxmoe_ref {
template <typename MatricesInfo, typename TileConfig,
          GroupScheduleMode GroupSchedule = GroupScheduleMode::kDeviceOnly>
void group_gemm_cutlass_f16(             //
    typename MatricesInfo::TA** ptr_As,  //
    typename MatricesInfo::TB** ptr_Bs,  //
    typename MatricesInfo::TC** ptr_Cs,  //
    typename MatricesInfo::TC** ptr_Ds,  //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    GemmCoord* h_problem_sizes,          //
    int problem_count                    //
) {
  using TA_ = typename MatricesInfo::TA;
  using TB_ = typename MatricesInfo::TB;
  using TC_ = typename MatricesInfo::TC;

  static_assert(std::is_same_v<TA_, half> && std::is_same_v<TB_, half> && std::is_same_v<TC_, half>,
                "Only support half precision now");

  using cutlass::half_t;
  using cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
  using T_ACC = float;

  using Epilogue = cutlass::epilogue::thread::LinearCombination<half_t, 128 / (sizeof(TC_) * 8), T_ACC, T_ACC>;

  constexpr int BM       = TileConfig::BM;
  constexpr int BN       = TileConfig::BN;
  constexpr int BK       = TileConfig::BK;
  constexpr int WARP_M   = TileConfig::WARP_M;
  constexpr int WARP_N   = TileConfig::WARP_N;
  constexpr int WARP_K   = TileConfig::WARP_K;
  constexpr int K_STAGE  = TileConfig::STAGE;
  using ThrBlockShape    = cutlass::gemm::GemmShape<BM, BN, BK>;
  using WarpShape        = cutlass::gemm::GemmShape<BM / WARP_M, BN / WARP_N, BK / WARP_K>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<          //
      half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,     //
      half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,  //
      half_t, cutlass::layout::RowMajor,                                          //
      T_ACC, cutlass::arch::OpClassTensorOp,                                      //
      cutlass::arch::Sm80,                                                        //
      ThrBlockShape,                                                              // CTA  Tile Shape
      WarpShape,                                                                  // Warp Tile Shape
      InstructionShape,                                                           // Inst Tile Shape
      Epilogue,                                                                   //
      GemmBatchedIdentityThreadblockSwizzle,                                      // CTA swizzle: no support now
      K_STAGE,                                                                    // Stage
      GroupSchedule                                                               // how to schedule the group gemm
      >::GemmKernel;

  auto epilogue_op = typename Epilogue::Params(1.0f, 0.0f);

  using GroupGemm     = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using GroupGemmArgs = typename GroupGemm::Arguments;
  int num_cta         = GroupGemm::sufficient(h_problem_sizes, problem_count);

  auto args = GroupGemmArgs(               //
      problem_sizes,                       // problem size for each gemm
      problem_count,                       // number of activated experts
      num_cta,                             //
      epilogue_op,                         //
      reinterpret_cast<half_t**>(ptr_As),  // expert ids:     [expert_freq, K]
      reinterpret_cast<half_t**>(ptr_Bs),  // expert weights: [N, K]
      reinterpret_cast<half_t**>(ptr_Cs),  // expert outputs: [expert_freq, N]
      reinterpret_cast<half_t**>(ptr_Ds),  //
      ldas,                                // [K]
      ldbs,                                // [K]
      ldcs,                                // [N]
      ldds,                                //
      h_problem_sizes                      //
  );

  auto kernel = GroupGemm{};

  size_t workspace_size = kernel.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(kernel.initialize(args, workspace.get()));
  CUTLASS_CHECK(kernel.run());
}

// Uniform symmetric W8A8. A is row-major MxK, B is column-major KxN, and C/D are
// row-major INT32. This does not dequantize.
inline void group_gemm_cutlass_w8a8(     //
    int8_t** ptr_As,                     //
    int8_t** ptr_Bs,                     //
    int32_t** ptr_Cs,                    //
    int32_t** ptr_Ds,                    //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    GemmCoord* h_problem_sizes,          //
    int problem_count                    //
) {
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int32_t;
  using ElementAccumulator = int32_t;
  constexpr int kAlignment = 16;

  using cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
  // Integer compute avoids the FP32 epilogue narrowing values above 2^24.
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<                 //
      ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlignment, //
      ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,          //
      kAlignment,                                                                        //
      ElementC, cutlass::layout::RowMajor,                                               //
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,           //
      cutlass::gemm::GemmShape<64, 128, 128>,                                            //
      cutlass::gemm::GemmShape<32, 64, 128>,                                             //
      cutlass::gemm::GemmShape<16, 8, 32>,                                               //
      Epilogue,                                                                          //
      GemmBatchedIdentityThreadblockSwizzle,                                             //
      3,                                                                                 //
      GroupScheduleMode::kDeviceOnly>::GemmKernel;

  using GroupGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  auto args       = typename GroupGemm::Arguments(                                 //
      problem_sizes,                                                               //
      problem_count,                                                               //
      GroupGemm::sufficient(h_problem_sizes, problem_count),                       //
      typename Epilogue::Params(ElementAccumulator(1), ElementAccumulator(0)),     //
      ptr_As,                                                                      //
      ptr_Bs,                                                                      //
      ptr_Cs,                                                                      //
      ptr_Ds,                                                                      //
      ldas,                                                                        //
      ldbs,                                                                        //
      ldcs,                                                                        //
      ldds,                                                                        //
      h_problem_sizes                                                              //
  );

  GroupGemm kernel;
  CUTLASS_CHECK(GroupGemm::can_implement(args));
  const size_t workspace_size = kernel.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(kernel.initialize(args, workspace.get()));
  CUTLASS_CHECK(kernel.run());
}

// Uniform symmetric W4A4. Pointers address CUTLASS packed int4b_t storage
// (two logical elements per byte, even element in the low nibble). Leading
// dimensions are logical elements, not compressed bytes. Output is INT32.
inline void group_gemm_cutlass_w4a4(     //
    cutlass::int4b_t** ptr_As,           //
    cutlass::int4b_t** ptr_Bs,           //
    int32_t** ptr_Cs,                    //
    int32_t** ptr_Ds,                    //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    GemmCoord* h_problem_sizes,          //
    int problem_count                    //
) {
  using ElementA           = cutlass::int4b_t;
  using ElementB           = cutlass::int4b_t;
  using ElementC           = int32_t;
  using ElementAccumulator = int32_t;
  constexpr int kAlignment = 32;

  using cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
  using Epilogue = cutlass::epilogue::thread::LinearCombination<
      ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<                 //
      ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlignment, //
      ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,          //
      kAlignment,                                                                        //
      ElementC, cutlass::layout::RowMajor,                                               //
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,           //
      cutlass::gemm::GemmShape<64, 128, 256>,                                            //
      cutlass::gemm::GemmShape<32, 64, 256>,                                             //
      cutlass::gemm::GemmShape<16, 8, 64>,                                               //
      Epilogue,                                                                          //
      GemmBatchedIdentityThreadblockSwizzle,                                             //
      3,                                                                                 //
      GroupScheduleMode::kDeviceOnly>::GemmKernel;

  using GroupGemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  auto args       = typename GroupGemm::Arguments(                                 //
      problem_sizes,                                                               //
      problem_count,                                                               //
      GroupGemm::sufficient(h_problem_sizes, problem_count),                       //
      typename Epilogue::Params(ElementAccumulator(1), ElementAccumulator(0)),     //
      ptr_As,                                                                      //
      ptr_Bs,                                                                      //
      ptr_Cs,                                                                      //
      ptr_Ds,                                                                      //
      ldas,                                                                        //
      ldbs,                                                                        //
      ldcs,                                                                        //
      ldds,                                                                        //
      h_problem_sizes                                                              //
  );

  GroupGemm kernel;
  CUTLASS_CHECK(GroupGemm::can_implement(args));
  const size_t workspace_size = kernel.get_workspace_size(args);
  cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(kernel.initialize(args, workspace.get()));
  CUTLASS_CHECK(kernel.run());
}

template <typename MatricesInfo>
void group_gemm_tn_cuda_ref(             //
    typename MatricesInfo::TA** ptr_As,  //
    typename MatricesInfo::TB** ptr_Bs,  //
    typename MatricesInfo::TC** ptr_Cs,  //
    typename MatricesInfo::TC** ptr_Ds,  //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    GemmCoord* h_problem_sizes,          //
    int problem_count                    //
) {
  using tiled_gemm::Layout;
  using TA = typename MatricesInfo::TA;
  using TB = typename MatricesInfo::TB;
  using TC = typename MatricesInfo::TC;

  using LayoutA = std::conditional_t<MatricesInfo::LAYOUT_A == Layout::RowMajor, cutlass::layout::RowMajor,
                                     cutlass::layout::ColumnMajor>;
  using LayoutB = std::conditional_t<MatricesInfo::LAYOUT_B == Layout::RowMajor, cutlass::layout::RowMajor,
                                     cutlass::layout::ColumnMajor>;
  using LayoutC = std::conditional_t<MatricesInfo::LAYOUT_C == Layout::RowMajor, cutlass::layout::RowMajor,
                                     cutlass::layout::ColumnMajor>;

  using cutlass::TensorRef, cutlass::make_TensorRef;
  using cutlass::layout::ColumnMajor;
  using cutlass::layout::RowMajor;
  using cutlass::reference::device::Gemm;

  using cutlass_gemm = Gemm<TA, LayoutA, TB, LayoutB, TC, LayoutC, float, float>;

  std::vector<half*> h_As(problem_count);
  std::vector<half*> h_Bs(problem_count);
  std::vector<half*> h_Cs(problem_count);

  cudaMemcpy(&h_As, ptr_As, problem_count * sizeof(TA*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_Bs, ptr_Bs, problem_count * sizeof(TB*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_Cs, ptr_Cs, problem_count * sizeof(TC*), cudaMemcpyDeviceToHost);

  for (int32_t i = 0; i < problem_count; ++i) {
    auto M = h_problem_sizes[i].m();
    auto N = h_problem_sizes[i].n();
    auto K = h_problem_sizes[i].k();
    // fmt::print("M: {}, N: {}, K: {}\n", M, N, K);

    auto ref_a = make_TensorRef(h_As[i], LayoutA(K));
    auto ref_b = make_TensorRef(h_Bs[i], LayoutB(K));
    auto ref_c = make_TensorRef(h_Cs[i], LayoutC(N));

    cutlass_gemm{}({M, N, K}, 1.0f, ref_a, ref_b, 0.0f, ref_c);
  }
}

struct ProblemVistor {
  int problem_count{};
  int num_group_tiles{};
  int* problem_tiles_prefix_sum{};
  GemmCoord* problem_sizes{};

  __host__ __device__ static auto get_group_tile_counts(GemmCoord* problem_sizes, int problem_count,
                                                        GemmCoord tile_size) {
    auto BM = tile_size.m();
    auto BN = tile_size.n();

    auto total_tiles = 0;
    for (int i = 0; i < problem_count; i++) {
      auto M = problem_sizes[i].m();
      auto N = problem_sizes[i].n();
      total_tiles += ceil_div(M, BM) * ceil_div(N, BN);
    }
    return total_tiles;
  }

  __host__ __device__ ProblemVistor(GemmCoord* _problem_sizes, int _problem_count, int* _problem_tiles_prefix_sum) {
    problem_tiles_prefix_sum = _problem_tiles_prefix_sum;
    num_group_tiles          = _problem_tiles_prefix_sum[_problem_count - 1];
    problem_count            = _problem_count;
    problem_sizes            = _problem_sizes;
  }

  /// if tile_idx > total_tiles, return -1
  /// else return the problem index
  __host__ __device__ auto get_problem_idx(int tile_idx) {
    auto problem_idx = -1;
    if (tile_idx >= num_group_tiles) return problem_idx;

    for (auto i = 0; i < problem_count; i++) {
      if (tile_idx >= problem_tiles_prefix_sum[i]) {
        continue;
      }
      problem_idx = i;
      break;
    }
    return problem_idx;
  }

  __host__ __device__ auto get_tile_coord(int problem_idx, int tile_idx, int BM, int BN) {
    auto problem_size = problem_sizes[problem_idx];
    auto M            = problem_size.m();
    auto N            = problem_size.n();

    auto problem_grid_shape = dim3(ceil_div(M, BM), ceil_div(N, BN), 1);

    if (problem_idx != 0) tile_idx -= problem_tiles_prefix_sum[problem_idx - 1];

    return dim3(tile_idx / problem_grid_shape.y, tile_idx % problem_grid_shape.y, 0);
  }
};

template <typename MatricesInfo, typename TileConfig>
__global__ void group_gemm_v1_impl(      //
    typename MatricesInfo::TA** ptr_As,  //
    typename MatricesInfo::TB** ptr_Bs,  //
    typename MatricesInfo::TC** ptr_Cs,  //
    typename MatricesInfo::TC** ptr_Ds,  //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    int problem_count,                   //
    int* problem_tiles_prefix_sum        //
) {
  using namespace tiled_gemm;

  using TA = typename MatricesInfo::TA;
  using TB = typename MatricesInfo::TB;
  using TC = typename MatricesInfo::TC;
  static_assert(std::is_same_v<TA, TB> && std::is_same_v<TA, half>, "");

  constexpr int BM = TileConfig::BM;
  constexpr int BN = TileConfig::BN;

  using TileA = GemmTileA<TA, MatricesInfo::LAYOUT_A, TileConfig>;
  using TileB = GemmTileB<TB, MatricesInfo::LAYOUT_B, TileConfig>;
  using TileC = GlobalTileC<TC, MatricesInfo::LAYOUT_C, TileConfig>;

  int lane   = threadIdx.x;
  int warp_x = TileConfig::warp_x();
  int warp_y = TileConfig::warp_y();
  int warp_z = TileConfig::warp_z();
  int tidx   = threadIdx.x + threadIdx.y * WarpSize;

  // allocate smem tile
  extern __shared__ uint8_t smem[];

  auto visitor = ProblemVistor(problem_sizes, problem_count, problem_tiles_prefix_sum);

  auto tile_idx = blockIdx.x;
  auto cta_size = gridDim.x;

  while (true) {
    // get corresponding [tileA, tileB, tileC] offset
    auto problem_idx = visitor.get_problem_idx(tile_idx);
    // early exit if all problems are done
    if (problem_idx == -1) break;
    // get coordinate of TileC of current problem
    auto tile_coord = visitor.get_tile_coord(problem_idx, tile_idx, BM, BN);

    auto problem_size = problem_sizes[problem_idx];
    auto M            = problem_size.m();
    auto N            = problem_size.n();
    auto K            = problem_size.k();

    auto tile_a = TileA(ptr_As[problem_idx], M, K, tile_coord.y, tile_coord.x, warp_y, warp_z, tidx);
    auto tile_b = TileB(ptr_Bs[problem_idx], N, K, tile_coord.y, tile_coord.x, warp_x, warp_z, tidx);
    auto tile_c = TileC(ptr_Cs[problem_idx], M, N, tile_coord.y, tile_coord.x, tidx);

    cta_gemm_multistage_v2<TileConfig>(tile_a, tile_b, tile_c, smem, K, lane, warp_x, warp_y, warp_z);

    // advance
    tile_idx += cta_size;
  }
}

template <typename MatricesInfo, typename TileConfig>
void my_group_gemm(                      //
    typename MatricesInfo::TA** ptr_As,  //
    typename MatricesInfo::TB** ptr_Bs,  //
    typename MatricesInfo::TC** ptr_Cs,  //
    typename MatricesInfo::TC** ptr_Ds,  //
    int64_t* ldas,                       //
    int64_t* ldbs,                       //
    int64_t* ldcs,                       //
    int64_t* ldds,                       //
    GemmCoord* problem_sizes,            //
    GemmCoord* h_problem_sizes,          //
    int problem_count                    //
) {
  using TA = typename MatricesInfo::TA;
  using TB = typename MatricesInfo::TB;
  using TC = typename MatricesInfo::TC;

  constexpr int BM = TileConfig::BM;
  constexpr int BN = TileConfig::BN;
  constexpr int BK = TileConfig::BK;

  auto problem_tiles_prefix_sum = new int[problem_count];
  auto total_tiles              = 0;
  for (int i = 0; i < problem_count; i++) {
    auto M = h_problem_sizes[i].m();
    auto N = h_problem_sizes[i].n();

    total_tiles += ceil_div(M, BM) * ceil_div(N, BN);
    problem_tiles_prefix_sum[i] = total_tiles;
  }
  int* d_problem_tiles_prefix_sum;
  checkCudaErrors(cudaMalloc(&d_problem_tiles_prefix_sum, sizeof(int) * problem_count));
  checkCudaErrors(cudaMemcpy(d_problem_tiles_prefix_sum, problem_tiles_prefix_sum, problem_count * sizeof(int),
                             cudaMemcpyHostToDevice));

  auto kernel = group_gemm_v1_impl<MatricesInfo, TileConfig>;

  // auto launch_cfg = get_launch_cfg_persistant<TileConfig>(total_tiles, kernel);
  size_t smem_size = 2 * TileConfig::STAGE * (BK * BM + BK * BN);
  smem_size        = std::max(smem_size, BM * BN * sizeof(typename TileConfig::MMA_T_ACC) * (TileConfig::WARP_K - 1));

  int dev_id{};
  int num_sm{};
  int num_warps = TileConfig::WARP_N * TileConfig::WARP_M * TileConfig::WARP_K;
  int num_threads{32 * num_warps};
  int max_active_blocks{};

  checkCudaErrors(cudaGetDevice(&dev_id));
  checkCudaErrors(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  // checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, num_threads, smem_size));

  int num_ctas = num_sm;
  // int num_ctas = num_sm * num_ctas_per_sm;

  auto launch_cfg = cudaLaunchConfig_t{
      .gridDim          = dim3(num_ctas),
      .blockDim         = dim3(32, num_warps),
      .dynamicSmemBytes = smem_size,
      .stream           = 0,  // use default stream
      .attrs            = nullptr,
      .numAttrs         = 0,  // no attributes
  };

  checkCudaErrors(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_cfg.dynamicSmemBytes));
  // fmt::println("launch config: grid: {}, block: {}, smem: {}", launch_cfg.gridDim, launch_cfg.blockDim,
  //              launch_cfg.dynamicSmemBytes);
  // fmt::println("problem_tiles_prefix_sum: {}", problem_tiles_prefix_sum);
  checkCudaErrors(cudaLaunchKernelEx(  //
      &launch_cfg, kernel,             //
      ptr_As, ptr_Bs, ptr_Cs, ptr_Ds,  //
      ldas, ldbs, ldcs, ldds,          //
      problem_sizes,                   //
      problem_count,                   //
      d_problem_tiles_prefix_sum       //
      ));

  delete[] problem_tiles_prefix_sum;
  checkCudaErrors(cudaFree(d_problem_tiles_prefix_sum));
}
}  // namespace mxmoe_ref