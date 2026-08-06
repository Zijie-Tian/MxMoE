#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "cuda_utils.cuh"
#include "helper.h"
#include "layout.cuh"
#include "quantize.cuh"

#if COMPUTE_ARCH == 89
#include <cuda_fp8.h>
#endif

namespace tiled_gemm {

enum class BlockSwizzle { None };
enum class Partition { DataParallel, StreamK };
enum class PackDim { MN, K };

template <                               //
    typename T_PACK_,                    // pack type
    bool SYM_,                           // symmetric/asymmetric quantization
    int QBITS_,                          // quantization bits
    int GSIZE_           = -1,           // `-1` represent per-channel(for weight)/per-token(for activation)
    PackDim PACK_DIM_    = PackDim::MN,  // for weight quantization, we pack along row
    bool USE_FP_         = false,        // FP8, FP4
    typename T_SCALE_ZP_ = half          //
    >
struct QConfig {
  static constexpr bool IS_QUANT = true;

  using T_PACK                  = T_PACK_;
  using T_SCALE_ZP              = T_SCALE_ZP_;
  static constexpr bool SYM     = SYM_;
  static constexpr int QBITS    = QBITS_;
  static constexpr int GSIZE    = GSIZE_;
  static constexpr int PACK_NUM = sizeof(T_PACK) * 8 / QBITS;
  static constexpr bool USE_FP  = USE_FP_;

  static constexpr PackDim PACK_DIM = PACK_DIM_;
};

using QCFG_W8A8      = QConfig<half, true, 8, -1, PackDim::K>;
using QCFG_W4A4      = QConfig<half, true, 4, -1, PackDim::K>;
using QCFG_W4A4_G128 = QConfig<half, true, 4, 128, PackDim::K>;

struct NO_QUANT {
  static constexpr bool IS_QUANT = false;
  static constexpr int QBITS     = 16;
};

template <typename TA_, Layout LAYOUT_A, typename TileConfig>
struct GemmTileA_ {
  using TA = TA_;
  static_assert(std::is_same_v<TA, half>, "Only half is supported for now");

  static constexpr int BM          = TileConfig::BM;
  static constexpr int BK          = TileConfig::BK;
  static constexpr int CTA_SIZE    = TileConfig::CTA_SIZE;
  static constexpr int mmaM        = TileConfig::mmaM;
  static constexpr int mmaK        = TileConfig::mmaK;
  static constexpr int WARP_TILE_M = TileConfig::WARP_TILE_M;
  static constexpr int WARP_TILE_K = TileConfig::WARP_TILE_K;
  static constexpr int WARP_FRAG_M = TileConfig::WARP_FRAG_M;
  static constexpr int FRAG_A_TH   = TileConfig::FRAG_A_TH;

  using FragA = uint32_t[WARP_FRAG_M][FRAG_A_TH];

  // convert to row_major in logic
  static constexpr int TILE_ROW = LAYOUT_A == Layout::RowMajor ? BM : BK;
  static constexpr int TILE_COL = LAYOUT_A == Layout::RowMajor ? BK : BM;

  // TODO: Now only `LDGSTS.128`(16 bytes async load) is supported
  static constexpr int ELE_PER_VEC  = 16 / sizeof(TA);
  static constexpr int VEC_PER_TILE = BM * BK / ELE_PER_VEC;

  static_assert(TILE_COL % ELE_PER_VEC == 0, "A: size of contigous dim must be multiple of `VEC_SIZE`(16)");
  static_assert(VEC_PER_TILE % CTA_SIZE == 0 || VEC_PER_TILE < CTA_SIZE, "A: gmem => smem can't be vectorized");

  static constexpr int G2S_ACTIVE_THREADS  = VEC_PER_TILE < CTA_SIZE ? VEC_PER_TILE : CTA_SIZE;
  static constexpr int G2S_LD_ROW_PER_ITER = G2S_ACTIVE_THREADS * ELE_PER_VEC / TILE_COL;

  // for smem swizzle
  static constexpr int VEC_PER_ROW   = TILE_COL / ELE_PER_VEC;
  static constexpr int BYTES_PER_ROW = TILE_COL * sizeof(TA);
  static constexpr int L2_PREFETCH   = BYTES_PER_ROW / TileConfig::WARP_K;
  static_assert(BYTES_PER_ROW == 64 || BYTES_PER_ROW % 128 == 0, "A: Failed to swizzle smem");

  /// smem swizzle: given coordinate (i, j) in terms of `VEC`, return the
  /// swizzled offset
  /// - if BYTES_PER_ROW == 64:       i * VEC_PER_ROW + (j ^ ((i / 2) % 4))
  /// - if BYTES_PER_ROW % 128 == 0:  i * VEC_PER_ROW + (j ^ (i % 8))
  using Smem = std::conditional_t<BYTES_PER_ROW == 64, smem_t<SwizzleMode::k64B>, smem_t<SwizzleMode::k128B>>;

  const TA* gmem_ptr{nullptr};

  int leading{};
  int cta_MAX_LOGIC_ROW{};
  // for g2s load
  int g2s_th_ld_row{};
  int g2s_th_ld_col{};
  bool pred_guard{true};
  // for swizzled smem r/w
  uint32_t smem_st_off{};
  uint32_t smem_ld_off{};
  uint32_t _smem_ld_off{};

  __device__ GemmTileA_(int m, int k, int bx, int by, int warp_y, int warp_z, int tidx) {
    int lane = (tidx % 32);
    uint32_t s2r_lane_ld_row{};
    uint32_t s2r_lane_ld_col{};
    if constexpr (LAYOUT_A == Layout::RowMajor) {
      leading = k;

      s2r_lane_ld_row = warp_y * WARP_TILE_M + lane % 16;
      s2r_lane_ld_col = warp_z * WARP_TILE_K / ELE_PER_VEC + lane / 16;
    } else {
      leading = m;

      s2r_lane_ld_row = warp_y * WARP_TILE_M + lane / 16 * 8;
      s2r_lane_ld_col = warp_z * WARP_TILE_K + lane % 16;
    }
    cta_MAX_LOGIC_ROW = m - by * BM;

    g2s_th_ld_row = tidx / VEC_PER_ROW;
    g2s_th_ld_col = tidx % VEC_PER_ROW;

    smem_st_off  = Smem::template get_permuted_offset<VEC_PER_ROW>(g2s_th_ld_row, g2s_th_ld_col);
    smem_ld_off  = Smem::template get_permuted_offset<VEC_PER_ROW>(s2r_lane_ld_row, s2r_lane_ld_col);
    _smem_ld_off = smem_ld_off;

    pred_guard = tidx < G2S_ACTIVE_THREADS;
  }

  __device__ __forceinline__ void advance_gmem() {
    if constexpr (LAYOUT_A == Layout::RowMajor) {
      gmem_ptr += BK;
    } else {
      gmem_ptr += BK * leading;
    }
  }

  __device__ __forceinline__ void load_frag(int kk, FragA frag, uint4* smem) {
    static_assert(std::is_same_v<TA, half>, "Only half is supported for now");
    uint32_t ld_off = smem_ld_off;
#pragma unroll
    for (auto i = 0; i < WARP_FRAG_M; i++) {
      if constexpr (LAYOUT_A == Layout::RowMajor) {
        ldmatrix_m8n8x4(frag[i], smem + ld_off);
        ld_off = Smem::template advance_offset_by_row<16, VEC_PER_ROW>(ld_off);
      } else {
        // ldmatrix_m8n8x4_trans(frag[i], smem + ld_off);
        static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported for now");
      }
    }
    if constexpr (LAYOUT_A == Layout::RowMajor) {
      smem_ld_off = Smem::template advance_offset_by_column<2>(smem_ld_off, kk);
    }
  }

  __device__ __forceinline__ void reset_frag_offset() {
    if constexpr (LAYOUT_A == Layout::RowMajor) {
      smem_ld_off = _smem_ld_off;
    } else {
      static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported for now");
    }
  }
};

template <typename TA_, Layout LAYOUT_A, typename TileConfig>
struct GemmTileAGatherLoad : GemmTileA_<TA_, LAYOUT_A, TileConfig> {
  using Base = GemmTileA_<TA_, LAYOUT_A, TileConfig>;
  using TA   = typename Base::TA;

  const int32_t* row_index;

  __device__ GemmTileAGatherLoad(             //
      const TA* A, const int32_t* row_index,  //
      int m, int k, int bx, int by,           //
      int warp_y, int warp_z, int tidx        //
      )
      : Base(m, k, bx, by, warp_y, warp_z, tidx) {  //
    static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported for now");

    // add cta bias
    this->g2s_th_ld_row = by * Base::BM + this->g2s_th_ld_row;
    this->gmem_ptr      = A;
    this->row_index     = row_index;
  }

  // Deal with tail block: avoid illegal memory access
  // Check each lane's M-loading dimension to determine whether illegal
  __device__ __forceinline__ void load_smem(uint4* smem, bool pred) {
    static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported in predicated load for now");
    static_assert(std::is_same_v<TA, half>, "Only half is supported for now");

    uint32_t st_off = Base::smem_st_off;
#pragma unroll
    for (int i = 0; i < Base::TILE_ROW; i += Base::G2S_LD_ROW_PER_ITER) {
      int logic_row = row_index[this->g2s_th_ld_row + i];
      int logic_col = this->g2s_th_ld_col * Base::ELE_PER_VEC;
      int ld_off    = logic_row * this->leading + logic_col;

      pred = pred && (logic_row < this->cta_MAX_LOGIC_ROW) && this->pred_guard;

      cp_async_cg_pred<Base::L2_PREFETCH>(smem + st_off, this->gmem_ptr + ld_off, pred);

      st_off = Base::Smem::template advance_offset_by_row<Base::G2S_LD_ROW_PER_ITER, Base::VEC_PER_ROW>(st_off);
    }
  }
};

template <typename TA_, Layout LAYOUT_A, typename TileConfig>
struct GemmTileA : GemmTileA_<TA_, LAYOUT_A, TileConfig> {
  using Base = GemmTileA_<TA_, LAYOUT_A, TileConfig>;
  using TA   = typename Base::TA;

  int32_t* row_index;

  __device__ GemmTileA(                 //
      const TA* A,                      //
      int m, int k, int bx, int by,     //
      int warp_y, int warp_z, int tidx  //
      )
      : Base(m, k, bx, by, warp_y, warp_z, tidx) {  //
    static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported for now");

    // add cta bias
    auto cta_g2s_row = by * Base::BM;
    this->gmem_ptr   = A + cta_g2s_row * Base::leading;
    ;
  }

  // Deal with tail block: avoid illegal memory access
  // Check each lane's M-loading dimension to determine whether illegal
  __device__ __forceinline__ void load_smem(uint4* smem, bool pred) {
    static_assert(LAYOUT_A == Layout::RowMajor, "Only RowMajor is supported in predicated load for now");
    static_assert(std::is_same_v<TA, half>, "Only half is supported for now");

    uint32_t st_off = Base::smem_st_off;
#pragma unroll
    for (int i = 0; i < Base::TILE_ROW; i += Base::G2S_LD_ROW_PER_ITER) {
      int logic_row = this->g2s_th_ld_row + i;
      int logic_col = this->g2s_th_ld_col * Base::ELE_PER_VEC;
      int ld_off    = logic_row * this->leading + logic_col;

      pred = pred && (logic_row < this->cta_MAX_LOGIC_ROW) && this->pred_guard;

      cp_async_cg_pred<Base::L2_PREFETCH>(smem + st_off, this->gmem_ptr + ld_off, pred);

      st_off = Base::Smem::template advance_offset_by_row<Base::G2S_LD_ROW_PER_ITER, Base::VEC_PER_ROW>(st_off);
    }
  }
};

template <typename TB_, Layout LAYOUT_B, typename TileConfig>
struct GemmTileB {
  using TB = TB_;
  static_assert(std::is_same_v<TB, half>, "Only half is supported for now");

  static constexpr int BN          = TileConfig::BN;
  static constexpr int BK          = TileConfig::BK;
  static constexpr int CTA_SIZE    = TileConfig::CTA_SIZE;
  static constexpr int mmaN        = TileConfig::mmaN;
  static constexpr int mmaK        = TileConfig::mmaK;
  static constexpr int WARP_TILE_N = TileConfig::WARP_TILE_N;
  static constexpr int WARP_TILE_K = TileConfig::WARP_TILE_K;
  static constexpr int WARP_FRAG_N = TileConfig::WARP_FRAG_N;
  static constexpr int FRAG_B_TH   = TileConfig::FRAG_B_TH;

  using FragB = uint32_t[WARP_FRAG_N][FRAG_B_TH];

  // convert to row_major in logic
  static constexpr int TILE_ROW = LAYOUT_B == Layout::RowMajor ? BK : BN;
  static constexpr int TILE_COL = LAYOUT_B == Layout::RowMajor ? BN : BK;

  // TODO: Now only `LDGSTS.128`(16 bytes async load) is supported
  static constexpr int ELE_PER_VEC  = 16 / sizeof(TB);
  static constexpr int VEC_PER_TILE = BN * BK / ELE_PER_VEC;

  static_assert(TILE_COL % ELE_PER_VEC == 0, "B: size of contigous dim must be multiple of `VEC_SIZE`(16)");
  static_assert(VEC_PER_TILE % CTA_SIZE == 0 || VEC_PER_TILE < CTA_SIZE, "B: gmem => smem can't be vectorized");

  static constexpr int G2S_ACTIVE_THREADS  = VEC_PER_TILE < CTA_SIZE ? VEC_PER_TILE : CTA_SIZE;
  static constexpr int G2S_LD_ROW_PER_ITER = G2S_ACTIVE_THREADS * ELE_PER_VEC / TILE_COL;

  // for smem swizzle
  static constexpr int VEC_PER_ROW   = TILE_COL / ELE_PER_VEC;
  static constexpr int BYTES_PER_ROW = TILE_COL * sizeof(TB);
  static constexpr int L2_PREFETCH   = BYTES_PER_ROW / TileConfig::WARP_K;
  static_assert(BYTES_PER_ROW == 64 || BYTES_PER_ROW % 128 == 0, "B: Failed to swizzle smem");
  using Smem = std::conditional_t<BYTES_PER_ROW == 64, smem_t<SwizzleMode::k64B>, smem_t<SwizzleMode::k128B>>;

  const TB* cta_B{nullptr};
  int leading{0};
  // for g2s load
  int g2s_th_ld_row{};
  int g2s_th_ld_col{};
  bool pred_guard{true};
  // for swizzled smem r/w
  uint32_t smem_st_off{};
  uint32_t smem_ld_off{};
  uint32_t _smem_ld_off{};

  __device__ GemmTileB(const TB* B, int n, int k, int bx, int by, int warp_x, int warp_z, int tidx) {
    int lane = (tidx % 32);
    uint32_t s2r_lane_ld_row;
    uint32_t s2r_lane_ld_col;
    if constexpr (LAYOUT_B == Layout::RowMajor) {
      static_assert(LAYOUT_B == Layout::ColMajor, "Only ColMajor is supported for now");
    } else {
      leading = k;
      cta_B   = B + bx * BN * leading;

      s2r_lane_ld_row = warp_x * WARP_TILE_N + lane / 16 * 8 + lane % 8;
      s2r_lane_ld_col = warp_z * WARP_TILE_K / ELE_PER_VEC + (lane / 8 % 2);
      smem_ld_off     = Smem::template get_permuted_offset<VEC_PER_ROW>(s2r_lane_ld_row, s2r_lane_ld_col);
      _smem_ld_off    = smem_ld_off;
    }
    g2s_th_ld_row = tidx / VEC_PER_ROW;
    g2s_th_ld_col = tidx % VEC_PER_ROW;

    smem_st_off = Smem::template get_permuted_offset<VEC_PER_ROW>(g2s_th_ld_row, g2s_th_ld_col);

    // pred_guard = (cur_col + g2s_th_col < n_) && (tidx < G2S_ACTIVE_THREADS);
    pred_guard = tidx < G2S_ACTIVE_THREADS;
  }

  __device__ __forceinline__ void load_smem(uint4* smem, bool pred) {
    static_assert(LAYOUT_B == Layout::ColMajor, "Only ColMajor is supported in predicated load for now");
    static_assert(std::is_same_v<TB, half>, "Only half is supported for now");
    int st_off = smem_st_off;
#pragma unroll
    for (int i = 0; i < TILE_ROW; i += G2S_LD_ROW_PER_ITER) {
      int logic_row = g2s_th_ld_row + i;
      int logic_col = g2s_th_ld_col * ELE_PER_VEC;
      int ld_off    = logic_row * leading + logic_col;
      pred          = pred && pred_guard;
      cp_async_cg_pred<L2_PREFETCH>(smem + st_off, cta_B + ld_off, pred);
      // cp_async_stream(smem + st_off, cta_B + ld_off);
      st_off = Smem::template advance_offset_by_row<G2S_LD_ROW_PER_ITER, VEC_PER_ROW>(st_off);
    }
  }

  __device__ __forceinline__ void load_frag(int kk, FragB frag, uint4* smem) {
    static_assert(LAYOUT_B == Layout::ColMajor, "Only ColMajor is supported in predicated load for now");
    static_assert(std::is_same_v<TB, half>, "Only half is supported for now");
    uint32_t ld_off = smem_ld_off;
#pragma unroll
    for (auto j = 0; j < WARP_FRAG_N; j += 2) {
      ldmatrix_m8n8x4(frag[j], smem + ld_off);
      ld_off = Smem::template advance_offset_by_row<16, VEC_PER_ROW>(ld_off);
    }
    if constexpr (LAYOUT_B == Layout::ColMajor) {
      smem_ld_off = Smem::template advance_offset_by_column<2>(smem_ld_off, kk);
    }
  }

  __device__ __forceinline__ void advance_gmem() {
    if constexpr (LAYOUT_B == Layout::RowMajor) {
      cta_B += BK * leading;
    } else {
      cta_B += BK;
    }
  }

  __device__ __forceinline__ void reset_frag_offset() {
    if constexpr (LAYOUT_B == Layout::RowMajor) {
      static_assert(LAYOUT_B == Layout::ColMajor, "Only ColMajor is supported for now");
    } else {
      smem_ld_off = _smem_ld_off;
    }
  }
};

template <typename TC, Layout LAYOUT_C, typename TileConfig>
struct GlobalTileC {
  static constexpr int BM = TileConfig::BM;
  static constexpr int BN = TileConfig::BN;

  static constexpr int mmaM        = TileConfig::mmaM;
  static constexpr int mmaN        = TileConfig::mmaN;
  static constexpr int WARP_FRAG_M = TileConfig::WARP_FRAG_M;
  static constexpr int WARP_FRAG_N = TileConfig::WARP_FRAG_N;
  static constexpr int FRAG_C_TH   = TileConfig::FRAG_C_TH;

  static constexpr int WARP_TILE_M = TileConfig::WARP_TILE_M;
  static constexpr int WARP_TILE_N = TileConfig::WARP_TILE_N;

  static constexpr int WARP_N = TileConfig::WARP_N;
  static constexpr int WARP_K = TileConfig::WARP_K;

  static constexpr int TILE_ROW    = LAYOUT_C == Layout::RowMajor ? BM : BN;
  static constexpr int TILE_COL    = LAYOUT_C == Layout::RowMajor ? BN : BM;
  static constexpr int ELE_PER_VEC = 16 / sizeof(TC);

  // for smem swizzle
  static constexpr int VEC_PER_ROW   = TILE_COL / ELE_PER_VEC;
  static constexpr int BYTES_PER_ROW = TILE_COL * sizeof(TC);
  using Smem = std::conditional_t<BYTES_PER_ROW == 64, smem_t<SwizzleMode::k64B>, smem_t<SwizzleMode::k128B>>;

  using T_ACC = typename TileConfig::MMA_T_ACC;

  using FragC = std::conditional_t<std::is_same_v<T_ACC, float>, float[WARP_FRAG_M * WARP_FRAG_N][FRAG_C_TH],
                                   int32_t[WARP_FRAG_M * WARP_FRAG_N][FRAG_C_TH]>;

  TC* cta_C{nullptr};
  int cta_MAX_LOGIC_COL{0};
  int cta_MAX_LOGIC_ROW{0};
  int leading;

  __device__ GlobalTileC(TC* C, int m, int n, int bx, int by, int bz = 0) {
    static_assert(LAYOUT_C == Layout::RowMajor, "Only RowMajor output is supported for now");

    if constexpr (LAYOUT_C == Layout::RowMajor) {
      auto cta_row      = by * BM;
      auto cta_col      = bx * BN;
      leading           = n;
      cta_MAX_LOGIC_ROW = m - cta_row;
      cta_MAX_LOGIC_COL = n - cta_col;
      cta_C             = C + cta_row * leading + cta_col;
    } else if constexpr (LAYOUT_C == Layout::ColMajor) {
      static_assert(LAYOUT_C == Layout::RowMajor, "Only RowMajor output is supported for now");
    } else {
    }
  }

  __device__ __forceinline__ static void clear_frag(FragC frag) {
#pragma unroll
    for (auto i = 0; i < WARP_FRAG_M * WARP_FRAG_N; i++) {
#pragma unroll
      for (auto j = 0; j < FRAG_C_TH; j++) {
        frag[i][j] = 0;
      }
    }
  }

  /// for wxax kernel
  __device__ __forceinline__ void load_scale(  //
      half scale_a[WARP_FRAG_M * 2],           //
      half scale_b[WARP_FRAG_N * 2],           //
      half* smem_scale_a,                      //
      half* smem_scale_b,                      //
      int lane, int warp_x, int warp_y         //
  ) {
    auto col = lane % 4;
    auto row = lane / 4;

#pragma unroll
    for (auto i = 0; i < WARP_FRAG_M; i++) {
      scale_a[i * 2]     = smem_scale_a[warp_y * WARP_TILE_M + i * mmaM + row];
      scale_a[i * 2 + 1] = smem_scale_a[warp_y * WARP_TILE_M + i * mmaM + row + 8];
    }
#pragma unroll
    for (auto j = 0; j < WARP_FRAG_N; j++) {
      scale_b[j * 2]     = smem_scale_b[warp_x * WARP_TILE_N + j * mmaN + col];
      scale_b[j * 2 + 1] = smem_scale_b[warp_x * WARP_TILE_N + j * mmaN + col + 1];
    }
  }

  /// for wxax kernel
  template <typename T>
  __device__ __forceinline__ void scale_frag(                 //
      const half r_sa[WARP_FRAG_M * 2],                       //
      const half r_sb[WARP_FRAG_N * 2],                       //
      int32_t frag_in[WARP_FRAG_M * WARP_FRAG_N][FRAG_C_TH],  //
      T frag_out[WARP_FRAG_M * WARP_FRAG_N][FRAG_C_TH]        //
  ) {
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "Only half/float is supported for now");
#pragma unroll
    for (auto i = 0; i < WARP_FRAG_M; i++) {
#pragma unroll
      for (auto j = 0; j < WARP_FRAG_N; j++) {
        auto frag_off = i * WARP_FRAG_N + j;

        // frag_c_out[frag_off][0] += __int2half_rn(frag_c[frag_off][0]) *
        // r_sa[i * 2] * r_sb[j * 2]; frag_c_out[frag_off][1] +=
        // __int2half_rn(frag_c[frag_off][1]) * r_sa[i * 2] * r_sb[j * 2 + 1];
        // frag_c_out[frag_off][2] += __int2half_rn(frag_c[frag_off][2]) *
        // r_sa[i * 2 + 1] * r_sb[j * 2]; frag_c_out[frag_off][3] +=
        // __int2half_rn(frag_c[frag_off][3]) * r_sa[i * 2 + 1] * r_sb[j * 2 +
        // 1];

        frag_out[frag_off][0] += T(frag_in[frag_off][0]) * T(r_sa[i * 2] * r_sb[j * 2]);
        frag_out[frag_off][1] += T(frag_in[frag_off][1]) * T(r_sa[i * 2] * r_sb[j * 2 + 1]);
        frag_out[frag_off][2] += T(frag_in[frag_off][2]) * T(r_sa[i * 2 + 1] * r_sb[j * 2]);
        frag_out[frag_off][3] += T(frag_in[frag_off][3]) * T(r_sa[i * 2 + 1] * r_sb[j * 2 + 1]);
      }
    }
  }

  template <typename T>
  __device__ __forceinline__ void reduce_slicek(T (&frag)[WARP_FRAG_M * WARP_FRAG_N][TileConfig::FRAG_C_TH],
                                                uint8_t* shmem, int lane, int warp_x, int warp_y, int warp_z) {
    if constexpr (WARP_K > 1) {
      // auto smem = reinterpret_cast<uint4*>(shmem);
      //     // reduce warp frag along k-dim
      //       constexpr int WARP_TILE_SIZE = WARP_TILE_M * WARP_TILE_N /
      //       ELE_PER_VEC;
      //       // print_frag_float<TileConfig>(frag[0], 0, 0, 0, 0, 0, "fragc");
      //       // print_frag_float<TileConfig>(frag[0], 0, 0, 0, 0, 1, "fragc");
      //       if (warp_z > 0) {
      //         // load from smem
      //         auto warp_base =
      //             smem + (warp_z - 1) * BM * BN / ELE_PER_VEC + warp_y *
      //             WARP_N * WARP_TILE_SIZE + warp_x * WARP_TILE_SIZE;
      //         int vec_id    = lane / 2;
      //         int vec_row   = vec_id / 8;
      //         int vec_col   = vec_id % 8;
      //         int st_offset = vec_row * 8 + (vec_row ^ vec_col);
      // #pragma unroll
      //         for (auto i = 0; i < WARP_FRAG_N * WARP_FRAG_M; i++) {
      //           reinterpret_cast<uint2*>(warp_base + st_offset)[lane % 2] =
      //           *reinterpret_cast<uint2*>(frag[i]); st_offset += 16;
      //           reinterpret_cast<uint2*>(warp_base + st_offset)[lane % 2] =
      //           *(reinterpret_cast<uint2*>(frag[i]) + 1); st_offset += 16;
      //         }
      //       }
      //       __syncthreads();
      //       if (warp_z == 0) {
      //         int vec_row   = lane / 8;
      //         int vec_col   = lane % 8;
      //         int ld_offset = vec_row * 8 + (vec_row ^ vec_col);
      //         // load from smem
      // #pragma unroll
      //         for (auto k = 0; k < WARP_K - 1; k++) {
      //           auto warp_base =
      //               smem + k * BM * BN / ELE_PER_VEC + warp_y * WARP_N *
      //               WARP_TILE_SIZE + warp_x * WARP_TILE_SIZE;
      // #pragma unroll
      //           for (auto i = 0; i < WARP_FRAG_N * WARP_FRAG_M; i++) {
      //             T_ACC R[2];
      //             ldmatrix_m8n8x2(reinterpret_cast<uint32_t(&)[2]>(R),
      //             warp_base + ld_offset);
      //             // if (i == 0) {
      //             //   print_frag_float<TileConfig>(R, 0, 0, 0, 0, 0, "load
      //             frag");
      //             // }
      //             frag[i][0] += R[0];
      //             frag[i][1] += R[1];
      //             ld_offset += 32;
      //           }
      //         }
      //       }

      using ele_ty = std::remove_cv_t<T>;
      static_assert(TileConfig::FRAG_C_TH == 4 || TileConfig::FRAG_C_TH == 2,
                    "Only type of 2|4 bytes is supported for now");
      using FETCH_TYPE = std::conditional_t<TileConfig::FRAG_C_TH == 2, uint32_t,
                                            std::conditional_t<TileConfig::FRAG_C_TH == 4, uint2, void>>;

      auto smem = reinterpret_cast<ele_ty*>(shmem);

      constexpr int WARP_TILE_SIZE = WARP_TILE_M * WARP_TILE_N;
      // print_frag_float<TileConfig>(frag[0], 0, 0, 0, 0, 0, "fragc");
      // print_frag_float<TileConfig>(frag[0], 0, 0, 0, 0, 1, "fragc");

      // TODO: swizzle
      if (warp_z > 0) {
        auto warp_base = smem + (warp_z - 1) * BM * BN + warp_y * WARP_N * WARP_TILE_SIZE + warp_x * WARP_TILE_SIZE;
        // store to smem
#pragma unroll
        for (auto i = 0; i < WARP_FRAG_M; i++) {
#pragma unroll
          for (auto j = 0; j < WARP_FRAG_N; j++) {
            auto frag_off = i * WARP_FRAG_N + j;
            auto st_idx   = i * (16 * 8) * WARP_FRAG_N + j * (16 * 8) + lane * 2;

            ((FETCH_TYPE*)(warp_base + st_idx))[0]         = ((FETCH_TYPE*)(frag[frag_off]))[0];
            ((FETCH_TYPE*)(warp_base + st_idx + 8 * 8))[0] = ((FETCH_TYPE*)(frag[frag_off]))[1];
          }
        }
      }
      __syncthreads();

      if (warp_z == 0) {
        // load from smem
#pragma unroll
        for (auto k = 0; k < WARP_K - 1; k++) {
          auto warp_base = smem + k * BM * BN + warp_y * WARP_N * WARP_TILE_SIZE + warp_x * WARP_TILE_SIZE;
#pragma unroll
          for (auto i = 0; i < WARP_FRAG_M; i++) {
#pragma unroll
            for (auto j = 0; j < WARP_FRAG_N; j++) {
              auto frag_off = i * WARP_FRAG_N + j;
              auto ld_idx   = i * (16 * 8) * WARP_FRAG_N + j * (16 * 8) + lane * 2;
              // TODO: fix bug when FRAG has half2 element
              if constexpr (TileConfig::FRAG_C_TH == 4) {
                frag[frag_off][0] += warp_base[ld_idx];
                frag[frag_off][1] += warp_base[ld_idx + 1];
                frag[frag_off][2] += warp_base[ld_idx + 8 * 8];
                frag[frag_off][3] += warp_base[ld_idx + 8 * 8 + 1];
              } else if constexpr (TileConfig::FRAG_C_TH == 2) {
                reinterpret_cast<half2&>(frag[frag_off][0]) += reinterpret_cast<half2&>(warp_base[ld_idx]);
                reinterpret_cast<half2&>(frag[frag_off][1]) += reinterpret_cast<half2&>(warp_base[ld_idx + 8 * 8]);
              }
            }
          }
        }
      }
    }
  }

  template <typename SRC_TYPE = T_ACC, typename FRAG>
  __device__ __forceinline__ void write_frag_to_gmem(FRAG frag, int lane, int warp_x, int warp_y, int warp_z,
                                                     uint8_t* smem = nullptr) {
    static_assert(LAYOUT_C == Layout::RowMajor, "Only RowMajor output is supported for now");
    auto warp_row = warp_y * WARP_TILE_M;
    auto warp_col = warp_x * WARP_TILE_N;

    if (warp_row >= cta_MAX_LOGIC_ROW || warp_col >= cta_MAX_LOGIC_COL) return;

    // if constexpr (WARP_K > 1) {
    //   reduce_slicek(frag, smem, lane, warp_x, warp_y, warp_z);
    // }

    // early exit
    if (warp_z > 0) return;

    auto warp_C = cta_C + warp_row * leading + warp_col;
#pragma unroll
    for (auto i = 0; i < WARP_FRAG_M; i++) {
      auto lane_row = i * 16 + lane / 4;
      if (warp_row + lane_row >= cta_MAX_LOGIC_ROW) break;
#pragma unroll
      for (auto j = 0; j < WARP_FRAG_N; j++) {
        auto lane_col = j * 8 + lane % 4 * 2;
        auto frag_off = i * WARP_FRAG_N + j;
        auto st_idx   = lane_row * leading + lane_col;

        if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<TC, float>) {
          ((float2*)(warp_C + st_idx))[0] = (*(float2*)(frag[frag_off]));
          if (warp_row + lane_row + 8 < cta_MAX_LOGIC_ROW) {
            ((float2*)(warp_C + st_idx + 8 * leading))[0] = (*(float2*)(frag[frag_off] + 2));
          }
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<TC, half>) {
          ((half2*)(warp_C + st_idx))[0] = __float22half2_rn(*(float2*)(frag[frag_off]));
          if (warp_row + lane_row + 8 < cta_MAX_LOGIC_ROW) {
            ((half2*)(warp_C + st_idx + 8 * leading))[0] = __float22half2_rn(*(float2*)(frag[frag_off] + 2));
          }
        } else if constexpr (std::is_same_v<SRC_TYPE, half2> && std::is_same_v<TC, half>) {
          ((half2*)(warp_C + st_idx))[0] = frag[frag_off][0];
          if (warp_row + lane_row + 8 < cta_MAX_LOGIC_ROW) {
            ((half2*)(warp_C + st_idx + 8 * leading))[0] = frag[frag_off][1];
          }
        } else if constexpr (std::is_same_v<SRC_TYPE, half> && std::is_same_v<TC, half>) {
          ((half2*)(warp_C + st_idx))[0] = *reinterpret_cast<half2*>(frag[frag_off] + 0);
          if (warp_row + lane_row + 8 < cta_MAX_LOGIC_ROW) {
            ((half2*)(warp_C + st_idx + 8 * leading))[0] = *reinterpret_cast<half2*>(frag[frag_off] + 1);
          }
        } else {
          static_assert(LAYOUT_C != LAYOUT_C, "not supported");
        }
      }
    }
  }
};

template <int mmaM_, int mmaN_, int mmaK_, typename T_INPUT_ = half, typename T_ACC_ = float>
struct MmaInst {
  static constexpr int mmaM = mmaM_;
  static constexpr int mmaN = mmaN_;
  static constexpr int mmaK = mmaK_;

  using T_INPUT = T_INPUT_;
  using T_ACC   = T_ACC_;

  __device__ __forceinline__ static void mma_instruction(uint32_t* RC, const uint32_t* RA, const uint32_t* RB) {
    if constexpr (mmaM == 16 && mmaN == 8 && mmaK == 16) {
      if constexpr (std::is_same_v<T_INPUT, half> && std::is_same_v<T_ACC, float>) {
        mma_m16n8k16_fp16fp32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
      } else if constexpr (std::is_same_v<T_INPUT, nv_bfloat16> && std::is_same_v<T_ACC, float>) {
        mma_m16n8k16_bf16fp32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
      } else if constexpr (std::is_same_v<T_INPUT, half> && std::is_same_v<T_ACC, half>) {
        mma_m16n8k16_fp16fp16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
      } else {
        static_assert(mmaM != mmaM, "not supported");
      }
    } else if constexpr (mmaM == 16 && mmaN == 16 && mmaK == 16) {
      if constexpr (std::is_same_v<T_INPUT, half> && std::is_same_v<T_ACC, float>) {
        mma_m16n16k16_fp16fp32(RC[0], RC[1], RC[2], RC[3], RC[4], RC[5], RC[6], RC[7], RA[0], RA[1], RA[2], RA[3],
                               RB[0], RB[1], RB[2], RB[3]);
      } else if constexpr (std::is_same_v<T_INPUT, half> && std::is_same_v<T_ACC, half>) {
        mma_m16n16k16_fp16fp16(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RB[2], RB[3]);
      } else {
      }
    } else if constexpr (mmaM == 16 && mmaN == 8 && mmaK == 32) {
      if constexpr (std::is_same_v<T_INPUT, int8_t> && std::is_same_v<T_ACC, int32_t>) {
        mma_m16n8k32_s8s32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
#if COMPUTE_ARCH == 89
      } else if constexpr (std::is_same_v<T_INPUT, __nv_fp8_e4m3> && std::is_same_v<T_ACC, float>) {
        // FP8 e4m3 MMA is Ada/Hopper (sm89+) only; not available on A100 sm80.
        mma_m16n8k32_e4m3fp32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
#endif
      } else {
        static_assert(mmaM != mmaM, "not supported");
      }
    } else if constexpr (mmaM == 16 && mmaN == 8 && mmaK == 64) {
      static_assert(std::is_same_v<T_INPUT, nv_precision::s4> && std::is_same_v<T_ACC, int32_t>, "");
      mma_m16n8k64_s4s32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
    } else if constexpr (mmaM == 16 && mmaN == 8 && mmaK == 256) {
      static_assert(std::is_same_v<T_INPUT, nv_precision::b1> && std::is_same_v<T_ACC, int32_t>, "");
      mma_m16n8k256_b1s32(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1]);
    } else if constexpr (mmaM == 16 && mmaN == 8 && mmaK == 8) {
      static_assert(mmaM != mmaM, "not supported");
    } else {
      static_assert(mmaM != mmaM, "not supported");
    }
  }
};

using MMA_FP16_FP16 = MmaInst<16, 8, 16, half, half>;
using MMA_FP16_FP32 = MmaInst<16, 8, 16, half, float>;
using MMA_BF16_FP32 = MmaInst<16, 8, 16, nv_bfloat16, float>;
using MMA_S8_K32    = MmaInst<16, 8, 32, int8_t, int32_t>;
using MMA_S4_K64    = MmaInst<16, 8, 64, nv_precision::s4, int32_t>;

#if COMPUTE_ARCH == 89
using MMA_FP8E4M3_K32 = MmaInst<16, 8, 16, __nv_fp8_e4m3, float>;
#endif

template <typename TA, Layout LAYOUT, typename TileConfig, typename QConfig>
struct QuantTileA {
  static_assert(LAYOUT == Layout::RowMajor, "Only RowMajor is supported for now");

  static constexpr int BM          = TileConfig::BM;
  static constexpr int BK          = TileConfig::BK;
  static constexpr int CTA_SIZE    = TileConfig::CTA_SIZE;
  static constexpr int mmaM        = TileConfig::mmaM;
  static constexpr int mmaK        = TileConfig::mmaK;
  static constexpr int WARP_TILE_M = TileConfig::WARP_TILE_M;
  static constexpr int WARP_TILE_K = TileConfig::WARP_TILE_K;
  static constexpr int WARP_FRAG_M = TileConfig::WARP_FRAG_N;
  static constexpr int FRAG_A_TH   = TileConfig::FRAG_A_TH;

  static constexpr int PACK_NUM       = QConfig::PACK_NUM;
  static constexpr int QBITS          = QConfig::QBITS;
  static constexpr int GSIZE          = QConfig::GSIZE;
  static constexpr bool SYM_QUANT     = QConfig::SYM;
  static constexpr PackDim PACK_DIM   = QConfig::PACK_DIM;
  static constexpr int CTA_SCALE_SIZE = SYM_QUANT ? BM : BM * 2;
  static constexpr int G2S_SCALE_SIZE = CTA_SCALE_SIZE * std::max(cu_cdiv(BK, GSIZE), 1);

  using T_PACK = typename QConfig::T_PACK;
  static_assert(std::is_same_v<T_PACK, TA>, "TA should be the same as T_PACK");
  static_assert(PackDim::K == PACK_DIM, "Only PackDim::K is supported for now");
  static_assert(SYM_QUANT, "Now only sym quant is supported for TileA");

  // convert to row_major in logic; TODO: now assume ColMajor
  static constexpr int TILE_ROW = PACK_DIM == PackDim::K ? BM : BM / PACK_NUM;
  static constexpr int TILE_COL = PACK_DIM == PackDim::K ? BK / PACK_NUM : BK;

  // TODO: Now only `LDGSTS.128`(16 bytes async load) is supported
  static constexpr int ELE_PER_VEC  = 16 / sizeof(T_PACK);
  static constexpr int VEC_PER_TILE = TILE_ROW * TILE_COL / ELE_PER_VEC;

  static_assert(TILE_COL % ELE_PER_VEC == 0, "QuantA: size of contigous dim must be multiple of `VEC_SIZE`(16)");
  static_assert(VEC_PER_TILE % CTA_SIZE == 0 || VEC_PER_TILE < CTA_SIZE, "QuantA: gmem => smem can't be vectorized");

  static constexpr int G2S_ACTIVE_THREADS  = VEC_PER_TILE < CTA_SIZE ? VEC_PER_TILE : CTA_SIZE;
  static constexpr int G2S_LD_ROW_PER_ITER = G2S_ACTIVE_THREADS * ELE_PER_VEC / TILE_COL;

  // for smem swizzle
  static constexpr int VEC_PER_ROW   = TILE_COL / ELE_PER_VEC;
  static constexpr int BYTES_PER_ROW = TILE_COL * sizeof(TA);
  static constexpr int L2_PREFETCH   = BYTES_PER_ROW / TileConfig::WARP_K;
  static_assert(BYTES_PER_ROW == 64 || BYTES_PER_ROW % 128 == 0, "QuantA: Failed to swizzle smem");

  using Smem = std::conditional_t<BYTES_PER_ROW == 64, smem_t<SwizzleMode::k64B>, smem_t<SwizzleMode::k128B>>;

  static constexpr int SCALE_ZP_PER_TH = SYM_QUANT ? WARP_FRAG_M : 2 * WARP_FRAG_M;
  // e.g. `PACK_NUM` == 8, then WARP_TILE_N must be a multiple of `mmaN *
  // PACK_NUM = 64`
  // static_assert(WARP_FRAG_M % PACK_NUM == 0, "WARP_FRAG_M % PACK_NUM == 0 not satisfied");

  using FragA = uint32_t[WARP_FRAG_M][FRAG_A_TH];

  const TA* cta_A{nullptr};
  int leading{0};
  int cta_MAX_LOGIC_ROW{0};
  // for g2s load
  int g2s_th_ld_row{0};
  int g2s_th_ld_col{0};
  bool pred_guard{true};
  // for swizzled smem r/w
  uint32_t smem_st_off{};
  uint32_t smem_ld_off{};
  uint32_t _smem_ld_off{};
  // for quant
  int gmem_scale_iter_size;

  __device__ QuantTileA(const TA* A, int m, int k, int bx, int by, int warp_y, int warp_z, int tidx, int bz = 0) {
    int lane = (tidx % 32);
    uint32_t s2r_lane_ld_row{};
    uint32_t s2r_lane_ld_col{};
    if constexpr (LAYOUT == Layout::RowMajor) {
      if constexpr (PACK_DIM == PackDim::K) {
        leading = k / PACK_NUM;
        cta_A   = A + by * BM * leading;
      } else {
        static_assert(PACK_DIM == PackDim::K, "Only RowMajor is supported for now");
      }
      s2r_lane_ld_row = warp_y * WARP_TILE_M + lane % 16;
      s2r_lane_ld_col = warp_z * WARP_TILE_K / ELE_PER_VEC + lane / 16;
    } else {
      static_assert(LAYOUT == Layout::RowMajor, "only support RowMajor now");
    }
    cta_MAX_LOGIC_ROW = m - by * BM;

    g2s_th_ld_row = tidx / VEC_PER_ROW;
    g2s_th_ld_col = tidx % VEC_PER_ROW;

    gmem_scale_iter_size = (SYM_QUANT ? 1 : 2) * m;

    smem_st_off  = Smem::template get_permuted_offset<VEC_PER_ROW>(g2s_th_ld_row, g2s_th_ld_col);
    smem_ld_off  = Smem::template get_permuted_offset<VEC_PER_ROW>(s2r_lane_ld_row, s2r_lane_ld_col);
    _smem_ld_off = smem_ld_off;

    pred_guard = tidx < G2S_ACTIVE_THREADS;
  }

  /// default: fetch 1 group of scale
  __device__ __forceinline__ void load_scale_zp_smem(half* smem, const half* scale_zp, int tidx, bool pred) {
    static_assert(SYM_QUANT, "only symmetric quant is supported for now");
    if (pred) {
#pragma unroll
      for (auto k = 0; k < G2S_SCALE_SIZE / CTA_SCALE_SIZE; k++) {
        auto gmem = scale_zp + k * gmem_scale_iter_size;
#pragma unroll
        for (auto i = 0; i < CTA_SCALE_SIZE; i += CTA_SIZE) {
          auto off = i + tidx;
          if (off < CTA_SCALE_SIZE) {
            smem[off + k * CTA_SCALE_SIZE] = gmem[off];
          }
        }
      }
    }
  }

  __device__ __forceinline__ void load_scale_zp_smem_align(half* smem, const half* scale_zp, int tidx, bool pred) {
    static_assert(CTA_SCALE_SIZE % 8 == 0, "CTA_SCALE_SIZE % 8 == 0 not satisfied");
#pragma unroll
    for (auto k = 0; k < G2S_SCALE_SIZE / CTA_SCALE_SIZE; k++) {
      auto gmem = scale_zp + k * gmem_scale_iter_size;
#pragma unroll
      for (int i = 0; i < CTA_SCALE_SIZE; i += CTA_SIZE * 8) {
        int off = i + tidx * 8;
        pred &= off < CTA_SCALE_SIZE;
        cp_async_cg_pred(smem + off + k * CTA_SCALE_SIZE, gmem + off, pred);
      }
    }
  }

  __device__ __forceinline__ void load_smem(uint4* smem, bool pred) {
    uint32_t st_off = smem_st_off;
#pragma unroll
    for (int i = 0; i < TILE_ROW; i += G2S_LD_ROW_PER_ITER) {
      int logic_row = g2s_th_ld_row + i;
      int logic_col = g2s_th_ld_col * ELE_PER_VEC;
      int ld_off    = logic_row * leading + logic_col;

      pred = pred && (logic_row < cta_MAX_LOGIC_ROW) && pred_guard;

      cp_async_cg_pred<L2_PREFETCH>(smem + st_off, cta_A + ld_off, pred);

      st_off = Smem::template advance_offset_by_row<G2S_LD_ROW_PER_ITER, VEC_PER_ROW>(st_off);
    }
  }

  template <typename T>
  __device__ __forceinline__ void load_frag(int k, T frag, const uint4* smem) {
    static_assert(LAYOUT == Layout::RowMajor, "Only RowMajor is supported for now");
    uint32_t ld_off = smem_ld_off;
    if constexpr (PACK_DIM == PackDim::K) {
#pragma unroll
      for (auto i = 0; i < WARP_FRAG_M; i++) {
        ldmatrix_m8n8x4(frag[i], smem + ld_off);
        ld_off = Smem::template advance_offset_by_row<16, VEC_PER_ROW>(ld_off);
      }
      smem_ld_off = Smem::template advance_offset_by_column<2>(this->smem_ld_off, k);
    } else {
      static_assert(PACK_DIM == PackDim::K, "Only PackDim::K is supported for now");
    }
  }

  __device__ __forceinline__ void advance_gmem() {
    if constexpr (LAYOUT == Layout::RowMajor) {
      cta_A += PACK_DIM == PackDim::K ? BK / PACK_NUM : BK;
    } else {
      static_assert(LAYOUT == Layout::RowMajor, "only support RowMajor now");
    }
  }

  __device__ __forceinline__ void reset_frag_offset() {
    if constexpr (LAYOUT == Layout::RowMajor) {
      smem_ld_off = _smem_ld_off;
    } else {
      static_assert(LAYOUT == Layout::RowMajor, "Only RowMajor is supported for now");
    }
  }
};

template <typename TB, Layout LAYOUT, typename TileConfig, typename QConfig>
struct QuantTileB {
  // static_assert(std::is_same_v<TB, half> == true, "Only half is supported for
  // now");
  static_assert(LAYOUT == Layout::ColMajor, "Only ColMajor is supported for now");

  static constexpr int BN          = TileConfig::BN;
  static constexpr int BK          = TileConfig::BK;
  static constexpr int CTA_SIZE    = TileConfig::CTA_SIZE;
  static constexpr int mmaN        = TileConfig::mmaN;
  static constexpr int mmaK        = TileConfig::mmaK;
  static constexpr int WARP_TILE_N = TileConfig::WARP_TILE_N;
  static constexpr int WARP_TILE_K = TileConfig::WARP_TILE_K;
  static constexpr int WARP_FRAG_N = TileConfig::WARP_FRAG_N;
  static constexpr int FRAG_B_TH   = TileConfig::FRAG_B_TH;

  static constexpr int PACK_NUM       = QConfig::PACK_NUM;
  static constexpr int QBITS          = QConfig::QBITS;
  static constexpr int GSIZE          = QConfig::GSIZE;
  static constexpr bool SYM_QUANT     = QConfig::SYM;
  static constexpr PackDim PACK_DIM   = QConfig::PACK_DIM;
  static constexpr int CTA_SCALE_SIZE = SYM_QUANT ? BN : BN * 2;
  static constexpr int G2S_SCALE_SIZE = CTA_SCALE_SIZE * std::max(cu_cdiv(BK, GSIZE), 1);

  using T_PACK = typename QConfig::T_PACK;
  static_assert(std::is_same_v<T_PACK, TB>, "TB should be the same as T_PACK");

  // convert to row_major in logic; TODO: now assume ColMajor
  static constexpr int TILE_ROW = PACK_DIM == PackDim::K ? BN : BN / PACK_NUM;
  static constexpr int TILE_COL = PACK_DIM == PackDim::K ? BK / PACK_NUM : BK;

  // TODO: Now only `LDGSTS.128` is supported
  static constexpr int ELE_PER_VEC  = 16 / sizeof(T_PACK);
  static constexpr int VEC_PER_TILE = TILE_ROW * TILE_COL / ELE_PER_VEC;

  static_assert(TILE_COL % ELE_PER_VEC == 0, "QuantB: size of contigous dim must be multiple of `VEC_SIZE`(16)");
  static_assert(VEC_PER_TILE % CTA_SIZE == 0 || VEC_PER_TILE < CTA_SIZE, "QuantB: gmem => smem can't be vectorized");

  static constexpr int G2S_ACTIVE_THREADS  = VEC_PER_TILE < CTA_SIZE ? VEC_PER_TILE : CTA_SIZE;
  static constexpr int G2S_LD_ROW_PER_ITER = G2S_ACTIVE_THREADS * ELE_PER_VEC / TILE_COL;

  // for smem swizzle
  static constexpr int VEC_PER_ROW   = TILE_COL / ELE_PER_VEC;
  static constexpr int BYTES_PER_ROW = TILE_COL * sizeof(TB);
  static_assert(BYTES_PER_ROW == 64 || BYTES_PER_ROW % 128 == 0, "QuantB: Failed to swizzle smem");
  static constexpr int L2_PREFETCH = BYTES_PER_ROW / TileConfig::WARP_K;
  using Smem = std::conditional_t<BYTES_PER_ROW == 64, smem_t<SwizzleMode::k64B>, smem_t<SwizzleMode::k128B>>;

  static constexpr int SCALE_ZP_PER_TH = SYM_QUANT ? WARP_FRAG_N : 2 * WARP_FRAG_N;

  // e.g. `PACK_NUM` == 8, then WARP_TILE_N must be a multiple of `mmaN *
  // PACK_NUM = 64`
  static_assert(PACK_DIM != PackDim::MN || WARP_FRAG_N % PACK_NUM == 0, "WARP_FRAG_N % PACK_NUM == 0 not satisfied");

  using FragB  = uint32_t[WARP_FRAG_N][FRAG_B_TH];
  using QFragB = uint32_t[WARP_FRAG_N / PACK_NUM][FRAG_B_TH];
  using QScale = half[SCALE_ZP_PER_TH];

  static constexpr auto dequant_frag = [] {
    if constexpr (PACK_DIM == PackDim::MN)
      return Converter::dequant_frag<WARP_FRAG_N, PACK_NUM, FRAG_B_TH, SYM_QUANT, QBITS, SCALE_ZP_PER_TH>;
    else
      return nullptr;
  }();

  const TB* cta_B{nullptr};
  int leading{0};
  // for g2s load
  int g2s_th_ld_row{};
  int g2s_th_ld_col{};
  bool pred_guard{true};
  // for swizzled smem r/w
  uint32_t smem_st_off{};
  uint32_t smem_ld_off{};
  uint32_t _smem_ld_off{};
  // for quant
  int gmem_scale_iter_size{};

  __device__ QuantTileB(const TB* B, int n, int k, int bx, int by, int warp_x, int warp_z, int tidx, int bz = 0) {
    int lane = (tidx % 32);
    uint32_t s2r_lane_ld_row;
    uint32_t s2r_lane_ld_col;
    if constexpr (LAYOUT == Layout::RowMajor) {
      static_assert(LAYOUT == Layout::ColMajor, "only support ColMajor now");
    } else {
      if constexpr (PACK_DIM == PackDim::MN) {
        leading = k;
        cta_B   = B + bx * BN / PACK_NUM * leading;
      } else {
        leading = k / PACK_NUM;
        cta_B   = B + bx * BN * leading;
      }

      s2r_lane_ld_row =
          warp_x * (PACK_DIM == PackDim::MN ? WARP_TILE_N / PACK_NUM : WARP_TILE_N) + lane / 16 * 8 + lane % 8;
      s2r_lane_ld_col = warp_z * WARP_TILE_K / ELE_PER_VEC + (lane / 8 % 2);
      smem_ld_off     = Smem::template get_permuted_offset<VEC_PER_ROW>(s2r_lane_ld_row, s2r_lane_ld_col);
      _smem_ld_off    = smem_ld_off;
    }
    g2s_th_ld_row = tidx / VEC_PER_ROW;
    g2s_th_ld_col = tidx % VEC_PER_ROW;

    gmem_scale_iter_size = (SYM_QUANT ? 1 : 2) * n;

    smem_st_off = Smem::template get_permuted_offset<VEC_PER_ROW>(g2s_th_ld_row, g2s_th_ld_col);

    pred_guard = tidx < G2S_ACTIVE_THREADS;
  }

  // assume scale_zp is half
  __device__ __forceinline__ void load_scale_zp_smem_align(half* smem, const half* scale_zp, int tidx, bool pred) {
#pragma unroll
    for (auto k = 0; k < G2S_SCALE_SIZE / CTA_SCALE_SIZE; k++) {
      auto gmem = scale_zp + k * gmem_scale_iter_size;
#pragma unroll
      for (int i = 0; i < CTA_SCALE_SIZE; i += CTA_SIZE * 8) {
        int off = i + tidx * 8;
        pred &= off < CTA_SCALE_SIZE;
        cp_async_cg_pred(smem + off + k * CTA_SCALE_SIZE, gmem + off, pred);
      }
    }
  }

  template <int FETCH_SIZE = G2S_SCALE_SIZE>
  __device__ __forceinline__ void load_scale_zp_smem(half* smem, const half* scale_zp, int tidx, bool pred) {
    if (pred) {
#pragma unroll
      for (auto k = 0; k < G2S_SCALE_SIZE / CTA_SCALE_SIZE; k++) {
        auto gmem = scale_zp + k * gmem_scale_iter_size;
#pragma unroll
        for (auto i = 0; i < CTA_SCALE_SIZE; i += CTA_SIZE) {
          auto off = i + tidx;
          if (off < CTA_SCALE_SIZE) {
            smem[off + k * CTA_SCALE_SIZE] = gmem[off];
          }
        }
      }
    }
  }

  __device__ __forceinline__ void load_smem(uint4* smem, bool pred) {
    int st_off = smem_st_off;
#pragma unroll
    for (int i = 0; i < TILE_ROW; i += G2S_LD_ROW_PER_ITER) {
      int logic_row = g2s_th_ld_row + i;
      int logic_col = g2s_th_ld_col * ELE_PER_VEC;
      int ld_off    = logic_row * leading + logic_col;
      pred          = pred && pred_guard;
      cp_async_cg_pred<L2_PREFETCH>(smem + st_off, cta_B + ld_off, pred);
      // cp_async_stream(smem + st_off, cta_B + ld_off);
      st_off = Smem::template advance_offset_by_row<G2S_LD_ROW_PER_ITER, VEC_PER_ROW>(st_off);
    }
  }

  __device__ __forceinline__ void load_frag_scale(QScale frag_scale, const half* smem_scale_zp, int lane, int warp_x) {
    // s2r scale-zp
    auto s2r_scale_lane_off = warp_x * WARP_TILE_N + lane / 4;
#pragma unroll
    for (auto i = 0; i < WARP_FRAG_N; i++) {
      if constexpr (SYM_QUANT) {
        frag_scale[i] = smem_scale_zp[s2r_scale_lane_off + i * 8];
      } else {
        *reinterpret_cast<half2*>(frag_scale + i * 2) =
            *reinterpret_cast<const half2*>(smem_scale_zp + (s2r_scale_lane_off + i * 8) * 2);
      }
    }
  }

  template <typename T>
  __device__ __forceinline__ void load_frag(int kk, T frag, const uint4* smem) {
    static_assert(LAYOUT == Layout::ColMajor, "only support ColMajor now");

    if constexpr (PACK_DIM == PackDim::MN) {
      uint32_t ld_off = smem_ld_off;
#pragma unroll
      for (auto j = 0; j < WARP_FRAG_N / PACK_NUM; j++) {
        ldmatrix_m8n8x2(frag[j], smem + ld_off);
        ld_off = Smem::template advance_offset_by_row<8, VEC_PER_ROW>(ld_off);
      }
      smem_ld_off = Smem::template advance_offset_by_column<2>(smem_ld_off, kk);

    } else {
      uint32_t ld_off = smem_ld_off;
#pragma unroll
      for (auto j = 0; j < WARP_FRAG_N; j += 2) {
        ldmatrix_m8n8x4(frag[j], smem + ld_off);
        ld_off = Smem::template advance_offset_by_row<16, VEC_PER_ROW>(ld_off);
      }
      smem_ld_off = Smem::template advance_offset_by_column<2>(smem_ld_off, kk);
    }
  }

  __device__ __forceinline__ void advance_gmem() {
    if constexpr (LAYOUT == Layout::RowMajor) {
      cta_B += BK * leading;
    } else {
      cta_B += PACK_DIM == PackDim::K ? BK / PACK_NUM : BK;
    }
  }

  __device__ __forceinline__ void reset_frag_offset() {
    if constexpr (LAYOUT == Layout::RowMajor) {
      static_assert(LAYOUT == Layout::ColMajor, "Only ColMajor is supported for now");
    } else {
      smem_ld_off = _smem_ld_off;
    }
  }
};

template <                                                 //
    int BM_, int BN_, int BK_,                             //
    int WARP_M_, int WARP_N_, int WARP_K_,                 //
    int STAGE_,                                            //
    int SPLIT_K_       = -1,                               //
    typename MmaInst_  = MmaInst<16, 8, 16, half, float>,  //
    typename QConfigA_ = NO_QUANT,                         //
    typename QConfigB_ = NO_QUANT                          //
    >
struct TileConfig {
  using MMA_INST            = MmaInst_;
  using MMA_T_INPUT         = typename MMA_INST::T_INPUT;
  using MMA_T_ACC           = typename MMA_INST::T_ACC;
  static constexpr int mmaM = MmaInst_::mmaM;
  static constexpr int mmaN = MmaInst_::mmaN;
  static constexpr int mmaK = MmaInst_::mmaK;

  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int BK = BK_;

  static constexpr int WARP_M = WARP_M_;
  static constexpr int WARP_N = WARP_N_;
  static constexpr int WARP_K = WARP_K_;

  static_assert((BM > 0) && (BM % mmaM == 0), "");
  static_assert((BN > 0) && (BN % mmaN == 0), "");
  static_assert((BK > 0) && (BK % mmaK == 0), "");
  static_assert(BM % (WARP_M * mmaM) == 0, "");
  static_assert(BN % (WARP_N * mmaN) == 0, "");
  static_assert(BK % (WARP_K * mmaK) == 0, "");
  static_assert(BK == 32 || BK == 64 || BK == 128 || BK == 256 || BK == 512, "");

  static constexpr int CTA_SIZE = WARP_M * WARP_N * WARP_K * WarpSize;
  static constexpr int STAGE    = STAGE_;
  static constexpr int SPLIT_K  = SPLIT_K_;

  static constexpr int WARP_TILE_M = BM / WARP_M;
  static constexpr int WARP_TILE_N = BN / WARP_N;
  static constexpr int WARP_TILE_K = BK / WARP_K;
  static constexpr int WARP_FRAG_M = WARP_TILE_M / mmaM;
  static constexpr int WARP_FRAG_N = WARP_TILE_N / mmaN;

  static constexpr int WARP_ITER_K = WARP_TILE_K / mmaK;

  // to handle s4 and u4 dtype, we use custom bitsof instead of sizeof
  static constexpr int FRAG_A_TH = mmaM * mmaK / WarpSize * bitsof<MMA_T_INPUT>() / 32;
  static constexpr int FRAG_B_TH = mmaN * mmaK / WarpSize * bitsof<MMA_T_INPUT>() / 32;
  static constexpr int FRAG_C_TH = mmaM * mmaN / WarpSize * bitsof<MMA_T_ACC>() / 32;

  /// for quantization
  using QConfigA = QConfigA_;
  using QConfigB = QConfigB_;

  static constexpr bool IS_A_QUANT = QConfigA::QBITS != 16;
  static constexpr bool IS_B_QUANT = QConfigB::QBITS != 16;

  static constexpr int SMEM_SIZE_SCALE_ZP_A = [] {
    if constexpr (IS_A_QUANT)
      return BM * (QConfigA::SYM ? 2 : 4);
    else
      return 0;
  }();
  static constexpr int SMEM_SIZE_SCALE_ZP_B = [] {
    if constexpr (IS_B_QUANT)
      return BN * (QConfigB::SYM ? 2 : 4);
    else
      return 0;
  }();

  template <bool Serpentine = false, typename REG_C, typename REG_AB>
  __device__ __forceinline__ static void mma_compute(      //
      REG_C c_frag[WARP_FRAG_M * WARP_FRAG_N][FRAG_C_TH],  //
      REG_AB a_frag[WARP_FRAG_M][FRAG_A_TH],               //
      REG_AB b_frag[WARP_FRAG_N][FRAG_B_TH]                //
  ) {
    static_assert(sizeof(REG_C) == 4 && sizeof(REG_AB) == 4, "");
#pragma unroll
    for (int i = 0; i < WARP_FRAG_M; i++) {
      auto RA = a_frag[i];
#pragma unroll
      for (int j = 0; j < WARP_FRAG_N; j++) {
        auto js = j;
        if constexpr (Serpentine) {
          js = (i & 1) ? WARP_FRAG_N - 1 - j : j;
        }
        auto RC = c_frag[i * WARP_FRAG_N + js];
        auto RB = b_frag[js];
        MMA_INST::mma_instruction(            //
            reinterpret_cast<uint32_t*>(RC),  //
            reinterpret_cast<uint32_t*>(RA),  //
            reinterpret_cast<uint32_t*>(RB)   //
        );
      }
    }
  }

  __device__ __forceinline__ static auto warp_x() { return threadIdx.y % WARP_N; }
  __device__ __forceinline__ static auto warp_y() { return (threadIdx.y / WARP_N) % WARP_M; }
  __device__ __forceinline__ static auto warp_z() { return threadIdx.y / (WARP_N * WARP_M); }
};
}  // namespace tiled_gemm