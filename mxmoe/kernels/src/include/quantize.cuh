#pragma once
#include <cuda_fp16.h>
#ifndef MXMOE_DEVICE_ONLY
#include <thrust/host_vector.h>
#endif

#include <cstdint>
#include <cstdio>

#include "cuda_utils.cuh"

#ifndef MXMOE_DEVICE_ONLY
using thrust::host_vector;
#endif

namespace mxmoe {

struct QParams {
  int2 qbits{make_int2(16, 16)};
  int gsize{-1};
  bool sym{false};

  QParams() {}
  QParams(int2 _qbits, int _gsize, bool _sym) : qbits(_qbits), gsize(_gsize), sym(_sym) {}

  bool operator==(const QParams& other) const {
    return qbits.x == other.qbits.x && qbits.y == other.qbits.y && gsize == other.gsize && sym == other.sym;
  }
};

}  // namespace mxmoe

struct Converter {
  // clang-format off
  /// convert 16 2bits-uint => 16 fp16
  static __device__ void cvt_f16x16_u2x16(uint32_t* res, const uint32_t* inp) {
    uint32_t* h                           = res;
    uint32_t const&  i4s                  = *inp;
    static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK    = 0x00030003;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;  // `1024`

    #pragma unroll
    for(auto i = 0; i < 8; i++){
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[i]) : "r"(i4s >> (2*i)), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MAGIC_NUM_0));
    }
  }

  template <int N>
  static __device__ __forceinline__ void cvt_f16_u2(uint32_t* res, const uint32_t* inp) {
    static_assert(N % 16 == 0, "N must be a multiple of 16");
#pragma unroll
    for (auto i = 0; i < N; i += 16) {
      cvt_f16x16_u2x16(res + i / 2, inp + i / 16);
    }
  }

  /// convert 8 4bits-uint => 8 fp16
  template<bool Signed = false>
  static __device__ void cvt_f16x8_u4x8(uint32_t* res, const uint32_t* inp) {
    // uint32_t* h                           = res;
    // uint32_t const&  i4s                  = *inp;
    // static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
    // static constexpr uint32_t BOT_MASK    = 0x000f000f;
    // static constexpr uint32_t TOP_MASK    = 0x00f000f0;
    // static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;  // `1024`
    // static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;  // `64`
    // // const uint32_t            top_i4s     = i4s >> 8;
    // uint32_t top_i4s = __byte_perm(i4s, 0, 0x4321);

    // //  i4s:  v0   v1   v2   v3   v4   v5   v6   v7
    // // (i4s & 0000_0000_0000_1111_0000_0000_0000_1111 | 0x64006400) => get v3, v7
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
    // // (i4s & 0000_0000_1111_0000_0000_0000_1111_0000 | 0x64006400) => get v2, v6
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
    // asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(MAGIC_NUM_0));
    // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(MAGIC_NUM_1));
    // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(MAGIC_NUM_0));
    // asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[3]) : "r"(h[3]), "r"(MAGIC_NUM_1));
    uint *h = reinterpret_cast<uint *>(res);
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = Signed ? 0x64076407 : 0x64006400;
    uint const& i4s = *inp;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
  }

  template <bool SYM_QUANT, int N>
  static __device__ __forceinline__ void cvt_f16_u4(uint32_t* res, const uint32_t* inp) {
    static_assert(N % 8 == 0, "N must be a multiple of 8");
#pragma unroll
    for (auto i = 0; i < N; i += 8) {
      cvt_f16x8_u4x8<SYM_QUANT>(res + i / 2, inp + i / 8);
    }
  }

  /// convert 4 8bits-uint => 4 fp16
  template<bool norm=true>
  static __device__ void cvt_f16x4_u8x4(uint32_t* dst, const uint32_t* inp) {
    uint32_t src = inp[0];
    static constexpr uint32_t f16_magic = 0x64000000;
    // 01234567 01234567
    // SEEEEEMM MMMMMMMM
    //      1MM XXXXXXXX
    // (1 + x/2^10) * 2^(e-15) -> e-15=10 -> e=25=16+8+1 -> 01100100b -> 0x64
    dst[0] = __byte_perm(src, f16_magic, 0x7170);
    dst[1] = __byte_perm(src, f16_magic, 0x7372);
    if constexpr (norm) {
        for (int i = 0; i < 4; ++i) {
          ((half*)dst)[i] -= __ushort_as_half(0x6400U);
        }
    }
  }
  // clang-format on

  template <int N>
  static __device__ __forceinline__ void cvt_f16_u8(uint32_t* res, const uint32_t* inp) {
    static_assert(N % 8 == 0, "N must be a multiple of 8");
#pragma unroll
    for (auto i = 0; i < N; i += 4) {
      cvt_f16x4_u8x4(res + i / 2, inp + i / 4);
    }
  }

  /// dequant `1` 16x8 fragB to `PACK_NUM` 16x8 fragB
  template <int WARP_FRAG_N, int PACK_NUM, int ELE_PER_TH, bool SYM_QUANT, int QBITS, int SCALE_ZP_SIZE,
            bool RAW_ZERO = false>
  static __device__ __forceinline__ void dequant_frag(              //
      uint32_t dequant[WARP_FRAG_N][ELE_PER_TH],                    //
      const uint32_t (&qfrag)[WARP_FRAG_N / PACK_NUM][ELE_PER_TH],  //
      const half scale_zp[SCALE_ZP_SIZE]                            //
  ) {
#pragma unroll
    for (auto j = 0; j < WARP_FRAG_N; j += PACK_NUM) {
      // convert the x-bits packed value to 16-bits half
      if constexpr (QBITS == 8) {
        Converter::cvt_f16_u8<8>(&(dequant[j][0]), qfrag[j / PACK_NUM]);
      } else if constexpr (QBITS == 4) {
        Converter::cvt_f16_u4<SYM_QUANT, 16>(&(dequant[j][0]), qfrag[j / PACK_NUM]);
      } else if constexpr (QBITS == 2) {
        Converter::cvt_f16_u2<32>(&(dequant[j][0]), qfrag[j / PACK_NUM]);
      } else {
        static_assert(QBITS != QBITS, "only support [2, 4, 8] bits dequantization now");
      }

#pragma unroll
      for (auto i = 0; i < PACK_NUM; i++) {
        half scale, zp;
        if constexpr (SYM_QUANT) {
          scale = scale_zp[j + i];
          zp    = half{};
        } else {
          scale = scale_zp[(j + i) * 2];
          zp    = scale_zp[(j + i) * 2 + 1];
        }

        // __syncthreads();
        // if (threadIdx.y == 0 && threadIdx.z == 0 && 0 == blockIdx.x && 0 == blockIdx.y) {
        //   if (threadIdx.x == 0)
        //     printf("deq-fragb-%d [bx: %d, by: %d, warp_x: %d, warp_y: %d]\n", i, blockIdx.x, blockIdx.y, threadIdx.y,
        //            threadIdx.z);
        //   __syncwarp();
        //   for (auto n = 0; n < 2; n++) {
        //     for (auto r : {0, 1, 2, 3}) {
        //       for (auto j = 0; j < 2; j++) {
        //         for (auto t = r; t < 32; t += 4) {
        //           if (threadIdx.x == t) {
        //             printf("%-9.3f, ", float(*(((half*)(&(dequant[i][n]))) + j)));
        //             if (t >= 28) printf("\n");
        //           }
        //           __syncwarp();
        //         }
        //       }
        //     }
        //   }
        // }
        // __syncthreads();

        // Default is q*s+zp. RAW_ZERO is (q-z)*s: q and z are exact in half, so q==z stays 0.
#pragma unroll
        for (auto x = 0; x < ELE_PER_TH; x++) {
          if constexpr (RAW_ZERO && !SYM_QUANT) {
            half2 q = reinterpret_cast<half2&>(dequant[j + i][x]);
            half2 diff = __hsub2(q, half2(zp, zp));
            reinterpret_cast<half2&>(dequant[j + i][x]) = __hmul2(diff, half2(scale, scale));
          } else {
            reinterpret_cast<half2&>(dequant[j + i][x]) =
                __hfma2(reinterpret_cast<half2&>(dequant[j + i][x]), half2(scale, scale), half2(zp, zp));
          }
        }

        // __syncthreads();
        // if (threadIdx.y == 0 && threadIdx.z == 0 && 0 == blockIdx.x && 0 == blockIdx.y) {
        //   if (threadIdx.x == 0)
        //     printf("scaled-deq-fragb-%d [bx: %d, by: %d, warp_x: %d, warp_y: %d]\n", i, blockIdx.x, blockIdx.y,
        //            threadIdx.y, threadIdx.z);
        //   for (auto n = 0; n < 2; n++) {
        //     for (auto r : {0, 1, 2, 3}) {
        //       for (auto j = 0; j < 2; j++) {
        //         for (auto t = r; t < 32; t += 4) {
        //           if (threadIdx.x == t) {
        //             printf("%-9.3f, ", float(*(((half*)(&(dequant[i][n]))) + j)));
        //             if (threadIdx.x >= 28) printf("\n");
        //           }
        //         }
        //       }
        //     }
        //   }
        // }
        // __syncthreads();
      }
    }
  }
};

/// now only support per-channel quantization
__global__ inline void quant_weight(const half* w, half* fake_quant, int* quant, half* scale_zp, int N, int K,
                                    int qbits, bool sym, int gsize = -1) {
  int lane = threadIdx.x;
  int bx   = blockIdx.x;

  // each warp is responsible for a column

  auto ele_per_th  = (K + 32 - 1) / 32;
  auto inp_warp    = w + bx * K;
  auto local_start = lane * ele_per_th;
  auto local_min   = __ushort_as_half((unsigned short)0x7C00U);
  auto local_max   = -__ushort_as_half((unsigned short)0x7C00U);

  auto fq_warp    = fake_quant + bx * K;
  auto quant_warp = quant + bx * K;

#pragma unroll
  for (auto i = local_start; i < local_start + ele_per_th; i++) {
    if (i < K) {
      local_min = cu_min(local_min, inp_warp[i]);
      local_max = cu_max(local_max, inp_warp[i]);
    }
  }

  local_min = warp_reduce<half, cu_min>(local_min);
  local_max = warp_reduce<half, cu_max>(local_max);

  auto lower = half(sym ? -((1 << (qbits - 1)) - 1) : 0);
  auto upper = half(sym ? (1 << (qbits - 1)) - 1 : (1 << qbits) - 1);

  auto scale = half{};
  auto zp    = half{};

  if (!sym) {
    zp    = local_min;
    scale = (local_max - local_min) / upper;
  } else {
    zp    = half{};
    scale = cu_max(__habs(local_min), __habs(local_max)) / upper;
  }
  scale = scale == half{} ? half{1} : scale;

  if (lane == 0) {
    if (!sym) {
      scale_zp[bx * 2]     = scale;
      scale_zp[bx * 2 + 1] = zp;
    } else {
      scale_zp[bx] = scale;
    }
  }

#pragma unroll
  for (auto i = local_start; i < local_start + ele_per_th; i++) {
    if (i < K) {
      auto q_val   = (inp_warp[i] - zp) / scale;
      auto rounded = __half2int_rn(cu_min(cu_max(q_val, lower), upper));

      quant_warp[i] = rounded;
      fq_warp[i]    = half(rounded) * scale + zp;
    }
  }
}

enum class PermuteMode {
  Row,
  None,
};

/// used to compute pre-pack indices
/// ori => `perm` => proj => desired
template <int N = 8, int PROJ_LEN = 8>
__host__ __device__ void compose_perm_indices(const int* proj, const int* desired, int* perm) {
  static_assert(N % PROJ_LEN == 0, "N must be a multiple of PROJ_LEN");
  for (auto i = 0; i < N; i += PROJ_LEN) {
    for (auto j = 0; j < PROJ_LEN; j++) {
      perm[proj[j] + i] = desired[i + j];
    }
  }
}

#ifndef MXMOE_DEVICE_ONLY
// assume K % gsize == 0
inline auto permute_scale(const host_vector<half>& scale, int N, int K, int gsize, int qbits, bool sym) {
  host_vector<half> res(scale.size());
  int n_group = K / gsize;
  for (auto i = 0; i < n_group; i++) {
    for (auto j = 0; j < N; j++) {
      if (sym) {
        res[i * N + j] = scale[i + j * n_group];
      } else {
        auto ld_off     = (i + j * n_group) * 2;
        auto st_off     = (i * N + j) * 2;
        res[st_off]     = scale[ld_off];
        res[st_off + 1] = scale[ld_off + 1];
      }
    }
  }
  return res;
}

// permute column-major weight [N, K]
inline auto permute_weight(const host_vector<int>& w, int N, int K, int qbits, PermuteMode p = PermuteMode::Row)
    -> host_vector<int> {
  const int PACK_NUM = 16 / qbits;

  if (N % PACK_NUM != 0) {
    throw std::runtime_error("dimension `N` not aligned with pack_num");
  }

  host_vector<int> res(w.size());

  if (p == PermuteMode::Row) {
    constexpr int FRAG_M = 16;
    constexpr int FRAG_N = 8;

    auto intermediate_perm_ = host_vector<int>(PACK_NUM * 4, 0);
    int* intermediate_perm  = intermediate_perm_.data();

    if (qbits == 8) {
      constexpr int dequant_proj[4]{1, 0, 3, 2};
      constexpr int desired_perm[8]{0, 2, 4, 6, 1, 3, 5, 7};
      compose_perm_indices<8, 4>(dequant_proj, desired_perm, intermediate_perm);
    } else if (qbits == 4) {
      constexpr int dequant_proj[8]{3, 7, 2, 6, 1, 5, 0, 4};
      constexpr int desired_perm[16]{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
      compose_perm_indices<16, 8>(dequant_proj, desired_perm, intermediate_perm);
    } else if (qbits == 2) {
      constexpr int dequant_proj[16]{7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8};
      constexpr int desired_perm[32]{0, 8,  16, 24, 1, 9,  17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
                                     4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};
      compose_perm_indices<32, 16>(dequant_proj, desired_perm, intermediate_perm);
    } else {
      throw std::runtime_error("not implemented");
    }
    // fmt::println("intermediate_perm: {}", intermediate_perm_);
    for (auto j = 0; j < N; j += PACK_NUM * FRAG_N) {
      for (auto i = 0; i < K; i += FRAG_M) {
        for (auto trav_j = 0; trav_j < 8; trav_j++) {
          for (auto trav_i = 0; trav_i < 8; trav_i += 2) {
            auto buff_val = host_vector<int>{};  // 4 * PACK_N
            auto buff_ptr = host_vector<int>{};  // 4 * PACK_N
            for (auto jj = 0; jj < FRAG_N; jj += 8) {
              for (auto ii = 0; ii < FRAG_M; ii += 8) {
                for (auto trav_ii = 0; trav_ii < 2; trav_ii++) {
                  for (auto frag_id = 0; frag_id < PACK_NUM; frag_id++) {
                    auto idx = i + ii + trav_i + trav_ii + (j + jj + frag_id * FRAG_N + trav_j) * K;
                    buff_val.push_back(w[idx]);
                    buff_ptr.push_back(idx);
                  }
                }
              }
            }
            for (auto x = 0; x < buff_val.size(); x++) {
              res[buff_ptr[x]] = buff_val[intermediate_perm[x]];
            }
          }
        }
      }
    }
  } else {
    throw std::runtime_error("not implemented");
  }

  return res;
}

inline auto pack_weightonly(const host_vector<int>& weight, int N, int K, int qbits, int sym = false)
    -> host_vector<half> {
  if (!(qbits == 8 || qbits == 4 || qbits == 2)) {
    throw std::runtime_error("only support [2, 4, 8] bits asym quantization");
  }
  const auto PACK_NUM = 16 / qbits;

  if (weight.size() % PACK_NUM != 0) {
    throw std::runtime_error("vec size not aligned with pack_num");
  }

  auto res = host_vector<half>(weight.size() / PACK_NUM);

  constexpr int FRAG_N = 8;
  for (auto j = 0; j < N; j += PACK_NUM * FRAG_N) {
    auto out_j = j / PACK_NUM;
    for (auto i = 0; i < K; i++) {
      for (auto jj = 0; jj < FRAG_N; jj++) {
        auto pack_val = uint16_t{};

        for (auto frag_id = 0; frag_id < PACK_NUM; frag_id++) {
          pack_val <<= qbits;

          auto value = weight[i + (j + frag_id * FRAG_N + jj) * K];
          if (sym) {
            int sign = weight[i + (j + frag_id * FRAG_N + jj) * K] < 0;
            // for fast dequant: shift the numeric range from [-max_int, max_int] => [0, 2 * max_int]
            pack_val |= value + ((1 << (qbits - 1)) - 1);
          } else {
            pack_val |= value;
          }
        }
        res[i + (out_j + jj) * K] = reinterpret_cast<half&>(pack_val);
      }
    }
  }

  return res;
}

/// pack as uint16_t
template <typename TARGET_TYPE>
inline auto pack_wxax(               //
    const host_vector<int>& act,     //
    const host_vector<int>& weight,  //
    int M, int N, int K, int qbits   //
) {
  if (!(qbits == 8 || qbits == 4)) {
    throw std::runtime_error("only support [4, 8] bits sym quantization");
  }
  const auto PACK_NUM = 16 / qbits;

  if (K % PACK_NUM != 0) {
    throw std::runtime_error("pack_dim not aligned with pack_num");
  }

  auto cvt_int_to_target = [](int val) {
    if constexpr (std::is_same_v<TARGET_TYPE, int8_t>) {
      int sign = val < 0;
      return uint8_t((sign << 7) | (val & 0b1111111));
    } else if constexpr (std::is_same_v<TARGET_TYPE, nv_precision::s4>) {
      int sign = val < 0;
      return uint8_t((sign << 3) | (val & 0b111));
    } else {
      //
    }
  };

  auto packed_dim  = K / PACK_NUM;
  auto pack_act    = host_vector<half>(act.size() / PACK_NUM);
  auto pack_weight = host_vector<half>(weight.size() / PACK_NUM);

  for (auto j = 0; j < K; j += PACK_NUM) {
    for (auto i = 0; i < M; i++) {
      uint16_t pack_val = 0;
      for (auto x = 0; x < PACK_NUM; x++) {
        pack_val <<= qbits;
        pack_val |= cvt_int_to_target(act[i * K + j + x]);
      }
      pack_act[i * packed_dim + j / PACK_NUM] = reinterpret_cast<half&>(pack_val);
    }
    for (auto i = 0; i < N; i++) {
      uint16_t pack_val = 0;
      for (auto x = 0; x < PACK_NUM; x++) {
        pack_val <<= qbits;
        pack_val |= cvt_int_to_target(weight[i * K + j + x]);
      }
      pack_weight[i * packed_dim + j / PACK_NUM] = reinterpret_cast<half&>(pack_val);
    }
  }

  return std::make_tuple(std::move(pack_act), std::move(pack_weight));
}
#endif  // MXMOE_DEVICE_ONLY