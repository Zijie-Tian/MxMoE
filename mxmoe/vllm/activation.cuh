#pragma once

// Dynamic per-token activation prep for mixed A4/A8/A16 MxMoE.
// Include after the Torch/CUDA headers. kernels.cu binds quantize_activations_out.
//
// inputs: FP16 [R, K], or [R, 2K] when silu (gate in [:, :K], up in [:, K:], vLLM silu_and_mul).
// offsets: GPU int32/int64 [E+1], sorted, offsets[0]=0, offsets[E]=R, empty experts allowed.
// activation_bits: GPU int32 [E], values 4/8/16. clips: FP32 [E, 2] = (negative, positive) factors.
// packed: int16 [R, stride_words], stride_words is K/4 (A4 only) or K/2 (A8 present).
//   A4 writes the first K/4 words, A8 the first K/2. High-first two's complement:
//   A4 nibbles [15:12],[11:8],[7:4],[3:0] hold codes in [-7, 7];
//   A8 bytes [15:8],[7:0] hold codes in [-127, 127]. A16 rows are not written.
// scales: FP32 [R], one symmetric per-token scale. sums: int32 [R, K/128], sum of those
//   signed codes. A16 rows leave packed, scales, and sums untouched.
// float_output: FP16 [R, K]. Non-silu does not write it (it may alias inputs).
//   silu writes FP16 SiLU(gate)*up on A16 rows only. Low-bit rows quantize that same
//   register-rounded value and do not store a float intermediate.
//
// No allocation, no host offset read, no device sync. Current stream. Graph-safe.
// One block per actual row; expert id is a binary search of the GPU offsets.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mxmoe_activation {

constexpr int kActGroup = 128;
constexpr int kActThreads = 128;

struct alignas(16) Half8 {
  half v[8];
};

__device__ __forceinline__ Half8 load_half8(const half* ptr) {
  Half8 out;
  memcpy(&out, ptr, sizeof(out));
  return out;
}

__device__ __forceinline__ void store_half8(half* ptr, Half8 value) {
  memcpy(ptr, &value, sizeof(value));
}

__device__ __forceinline__ int16_t bits_as_i16(uint16_t word) {
  int16_t out;
  memcpy(&out, &word, sizeof(out));
  return out;
}

// vLLM silu_and_mul: round SiLU(gate) to FP16, multiply by up, round to FP16.
__device__ __forceinline__ half silu_mul_fp16(half gate, half up) {
  const float g = __half2float(gate);
  const float silu = g / (1.0f + expf(-g));
  const float y = __half2float(__float2half_rn(silu)) * __half2float(up);
  return __float2half_rn(y);
}

template <bool Silu>
__device__ __forceinline__ float source_at(const half* row, int K, int k) {
  if constexpr (Silu) {
    return __half2float(silu_mul_fp16(row[k], row[K + k]));
  } else {
    return __half2float(row[k]);
  }
}

__device__ __forceinline__ uint16_t pack_high_first_4(int q0, int q1, int q2, int q3) {
  return static_cast<uint16_t>(((static_cast<unsigned>(q0) & 0xFu) << 12) |
                               ((static_cast<unsigned>(q1) & 0xFu) << 8) |
                               ((static_cast<unsigned>(q2) & 0xFu) << 4) |
                               (static_cast<unsigned>(q3) & 0xFu));
}

__device__ __forceinline__ uint16_t pack_high_first_8(int q0, int q1) {
  return static_cast<uint16_t>(((static_cast<unsigned>(q0) & 0xFFu) << 8) |
                               (static_cast<unsigned>(q1) & 0xFFu));
}

// Clamp to the integer codebook, then round-to-nearest-even. Equivalent to
// round-then-clamp for integer qmax, and does not overflow int32 on FP16 inputs.
__device__ __forceinline__ int quantize_code(float x, float scale, int qmax) {
  const float limit = static_cast<float>(qmax);
  const float scaled = fminf(fmaxf(x / scale, -limit), limit);
  return __float2int_rn(scaled);
}

template <typename OffsetT>
__device__ int find_expert(const OffsetT* __restrict__ offsets, int E, int row) {
  int lo = 0;
  int hi = E;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (static_cast<int64_t>(__ldg(offsets + mid + 1)) <= static_cast<int64_t>(row)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <bool Silu>
__device__ void write_silu_row(const half* __restrict__ row, half* __restrict__ dst, int K) {
  const bool vec = ((reinterpret_cast<uintptr_t>(row) | reinterpret_cast<uintptr_t>(dst)) & 15u) == 0;
  if (vec) {
    const int nvec = K >> 3;
    for (int v = static_cast<int>(threadIdx.x); v < nvec; v += kActThreads) {
      const Half8 gate = load_half8(row + static_cast<int64_t>(v) * 8);
      const Half8 up = load_half8(row + K + static_cast<int64_t>(v) * 8);
      Half8 out;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        out.v[j] = silu_mul_fp16(gate.v[j], up.v[j]);
      }
      store_half8(dst + static_cast<int64_t>(v) * 8, out);
    }
  } else {
    for (int k = static_cast<int>(threadIdx.x); k < K; k += kActThreads) {
      dst[k] = silu_mul_fp16(row[k], row[K + k]);
    }
  }
}

template <bool Silu>
__device__ void row_extents(const half* __restrict__ row, int K, float& neg, float& pos) {
  neg = 0.f;
  pos = 0.f;
  const bool vec = (reinterpret_cast<uintptr_t>(row) & 15u) == 0;
  if (vec) {
    const int nvec = K >> 3;
    for (int v = static_cast<int>(threadIdx.x); v < nvec; v += kActThreads) {
      const Half8 gate = load_half8(row + static_cast<int64_t>(v) * 8);
      Half8 up;
      if constexpr (Silu) {
        up = load_half8(row + K + static_cast<int64_t>(v) * 8);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float x = Silu ? __half2float(silu_mul_fp16(gate.v[j], up.v[j])) : __half2float(gate.v[j]);
        neg = fmaxf(neg, -fminf(x, 0.f));
        pos = fmaxf(pos, fmaxf(x, 0.f));
      }
    }
  } else {
    for (int k = static_cast<int>(threadIdx.x); k < K; k += kActThreads) {
      const float x = source_at<Silu>(row, K, k);
      neg = fmaxf(neg, -fminf(x, 0.f));
      pos = fmaxf(pos, fmaxf(x, 0.f));
    }
  }
}

template <typename OffsetT, bool Silu>
__global__ void __launch_bounds__(kActThreads)
    mxmoe_quantize_activations(const half* __restrict__ inputs, const OffsetT* __restrict__ offsets,
                               const int* __restrict__ activation_bits, const float* __restrict__ clips,
                               int16_t* __restrict__ packed, float* __restrict__ scales, int32_t* __restrict__ sums,
                               half* __restrict__ float_output, int R, int K, int E, int stride_words) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= R) return;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t prev = static_cast<int64_t>(__ldg(offsets));
    assert(prev >= 0 && prev <= static_cast<int64_t>(R));
    for (int e = 1; e <= E; ++e) {
      const int64_t cur = static_cast<int64_t>(__ldg(offsets + e));
      assert(cur >= prev && cur <= static_cast<int64_t>(R));
      prev = cur;
    }
  }
  const int64_t first = static_cast<int64_t>(__ldg(offsets));
  const int64_t last = static_cast<int64_t>(__ldg(offsets + E));
  if (static_cast<int64_t>(row) < first || static_cast<int64_t>(row) >= last) return;

  const int expert = find_expert(offsets, E, row);
  if (expert >= E) {
    assert(false);
    return;
  }
  const int64_t start = static_cast<int64_t>(__ldg(offsets + expert));
  const int64_t end = static_cast<int64_t>(__ldg(offsets + expert + 1));
  if (!(start >= 0 && start <= static_cast<int64_t>(row) && static_cast<int64_t>(row) < end && end <= static_cast<int64_t>(R))) {
    assert(false);
    return;
  }

  const int bits = __ldg(activation_bits + expert);
  const float clip_neg = __ldg(clips + static_cast<int64_t>(expert) * 2);
  const float clip_pos = __ldg(clips + static_cast<int64_t>(expert) * 2 + 1);
  assert(bits == 4 || bits == 8 || bits == 16);
  assert(isfinite(clip_neg) && isfinite(clip_pos) && clip_neg > 0.f && clip_neg <= 1.f && clip_pos > 0.f &&
         clip_pos <= 1.f);

  const int64_t in_stride = Silu ? static_cast<int64_t>(K) * 2 : static_cast<int64_t>(K);
  const half* row_in = inputs + static_cast<int64_t>(row) * in_stride;

  if (bits == 16) {
    if constexpr (Silu) {
      write_silu_row<Silu>(row_in, float_output + static_cast<int64_t>(row) * K, K);
    }
    return;
  }

  assert((bits == 4 && stride_words >= K / 4) || (bits == 8 && stride_words >= K / 2));

  __shared__ float red_neg[kActThreads];
  __shared__ float red_pos[kActThreads];
  __shared__ float scale_s;
  __shared__ int warp_sums[4];

  float neg = 0.f;
  float pos = 0.f;
  row_extents<Silu>(row_in, K, neg, pos);
  red_neg[threadIdx.x] = neg;
  red_pos[threadIdx.x] = pos;
  __syncthreads();
  for (int stride = kActThreads / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      red_neg[threadIdx.x] = fmaxf(red_neg[threadIdx.x], red_neg[threadIdx.x + stride]);
      red_pos[threadIdx.x] = fmaxf(red_pos[threadIdx.x], red_pos[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const int qmax = bits == 4 ? 7 : 127;
  if (threadIdx.x == 0) {
    const float range = fmaxf(red_neg[0] * clip_neg, red_pos[0] * clip_pos);
    const float scale = (range > 0.f && isfinite(range)) ? range / static_cast<float>(qmax) : 1.f;
    scale_s = scale;
    scales[row] = scale;
  }
  __syncthreads();
  const float scale = scale_s;

  int16_t* packed_row = packed + static_cast<int64_t>(row) * stride_words;
  int32_t* sum_row = sums + static_cast<int64_t>(row) * (K / kActGroup);
  const int groups = K / kActGroup;
  for (int g = 0; g < groups; ++g) {
    const int k = g * kActGroup + static_cast<int>(threadIdx.x);
    const int q = quantize_code(source_at<Silu>(row_in, K, k), scale, qmax);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int q1 = __shfl_down_sync(0xffffffff, q, 1);
    if (bits == 4) {
      const int q2 = __shfl_down_sync(0xffffffff, q, 2);
      const int q3 = __shfl_down_sync(0xffffffff, q, 3);
      if ((lane & 3) == 0) {
        packed_row[static_cast<int64_t>(g) * 32 + (static_cast<int>(threadIdx.x) >> 2)] =
            bits_as_i16(pack_high_first_4(q, q1, q2, q3));
      }
    } else if ((lane & 1) == 0) {
      packed_row[static_cast<int64_t>(g) * 64 + (static_cast<int>(threadIdx.x) >> 1)] =
          bits_as_i16(pack_high_first_8(q, q1));
    }

    int sum = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (lane == 0) {
      warp_sums[threadIdx.x >> 5] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      sum_row[g] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    }
    __syncthreads();
  }
}

template <typename OffsetT, bool Silu>
void launch_quantize(const half* inputs, const OffsetT* offsets, const int* bits, const float* clips, int16_t* packed,
                     float* scales, int32_t* sums, half* float_output, int R, int K, int E, int stride_words,
                     cudaStream_t stream) {
  if (R == 0) return;
  mxmoe_quantize_activations<OffsetT, Silu><<<R, kActThreads, 0, stream>>>(
      inputs, offsets, bits, clips, packed, scales, sums, float_output, R, K, E, stride_words);
  C10_CUDA_CHECK(cudaGetLastError());
}

void check_type(const torch::Tensor& tensor, at::ScalarType type, const char* name, const char* what) {
  TORCH_CHECK(tensor.scalar_type() == type, name, " must be ", what, ", got ", tensor.scalar_type());
}

void check_rank(const torch::Tensor& tensor, int64_t rank, const char* name) {
  TORCH_CHECK(tensor.dim() == rank, name, " must be rank ", rank, ", got ", tensor.sizes());
}

bool storage_overlaps(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.defined() || !b.defined() || a.numel() == 0 || b.numel() == 0) return false;
  const auto* a0 = static_cast<const char*>(a.data_ptr());
  const auto* b0 = static_cast<const char*>(b.data_ptr());
  const auto* a1 = a0 + a.numel() * a.element_size();
  const auto* b1 = b0 + b.numel() * b.element_size();
  return a0 < b1 && b0 < a1;
}

}  // namespace mxmoe_activation

inline void quantize_activations_out(torch::Tensor inputs, torch::Tensor offsets, torch::Tensor activation_bits,
                                     torch::Tensor clips, torch::Tensor packed, torch::Tensor scales, torch::Tensor sums,
                                     torch::Tensor float_output, bool silu) {
  using namespace mxmoe_activation;
  const NamedTensor act_tensors[] = {
      {&inputs, "inputs"}, {&offsets, "offsets"}, {&activation_bits, "activation_bits"}, {&clips, "clips"},
      {&packed, "packed"}, {&scales, "scales"}, {&sums, "sums"}, {&float_output, "float_output"}};
  check_contiguous_then_device<true>(inputs, act_tensors);
  check_type(inputs, torch::kFloat16, "inputs", "fp16");
  check_type(float_output, torch::kFloat16, "float_output", "fp16");
  check_type(activation_bits, torch::kInt32, "activation_bits", "int32");
  check_type(clips, torch::kFloat, "clips", "float32");
  check_type(packed, torch::kInt16, "packed", "int16");
  check_type(scales, torch::kFloat, "scales", "float32");
  check_type(sums, torch::kInt32, "sums", "int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt || offsets.scalar_type() == torch::kLong,
              "offsets must be int32 or int64, got ", offsets.scalar_type());
  check_rank(inputs, 2, "inputs");
  check_rank(float_output, 2, "float_output");
  check_rank(offsets, 1, "offsets");
  check_rank(activation_bits, 1, "activation_bits");
  check_rank(clips, 2, "clips");
  check_rank(packed, 2, "packed");
  check_rank(scales, 1, "scales");
  check_rank(sums, 2, "sums");

  const int64_t R = inputs.size(0);
  const int64_t input_k = inputs.size(1);
  TORCH_CHECK(R >= 0, "inputs rows are negative");
  TORCH_CHECK(R <= std::numeric_limits<int>::max(), "routed rows exceed int32");
  int64_t K = input_k;
  if (silu) {
    TORCH_CHECK(input_k % 2 == 0, "silu inputs must be [R, 2K], got Kdim=", input_k);
    K = input_k / 2;
    TORCH_CHECK(!storage_overlaps(inputs, float_output),
                "silu float_output overlaps inputs; A16 rows write SiLU(gate)*up while reading both halves");
  }
  TORCH_CHECK(K > 0 && K % kActGroup == 0, "K=", K, " must be a positive multiple of ", kActGroup);
  TORCH_CHECK(K <= std::numeric_limits<int>::max(), "K exceeds int32");
  const int64_t in_stride = silu ? K * 2 : K;
  TORCH_CHECK(in_stride == 0 || R <= std::numeric_limits<int64_t>::max() / in_stride,
              "activation span ", R, "*", in_stride, " exceeds int64");

  const int64_t E = activation_bits.size(0);
  TORCH_CHECK(E >= 0 && E <= 65535, "E must be in [0, 65535], got ", E);
  TORCH_CHECK(offsets.numel() == E + 1, "offsets must be [E+1]=[", E + 1, "], got ", offsets.sizes());
  TORCH_CHECK(clips.sizes() == torch::IntArrayRef({E, 2}), "clips must be [E, 2]=[", E, ", 2], got ", clips.sizes());
  TORCH_CHECK(float_output.size(0) == R && float_output.size(1) == K,
              "float_output must be [R, K]=[", R, ", ", K, "], got ", float_output.sizes());

  const int64_t words_a4 = K / 4;
  const int64_t words_a8 = K / 2;
  const int64_t groups = K / kActGroup;
  TORCH_CHECK(packed.size(0) == R, "packed rows ", packed.size(0), " != R ", R);
  TORCH_CHECK(packed.size(1) == words_a4 || packed.size(1) == words_a8,
              "packed stride must be K/4=", words_a4, " or K/2=", words_a8, ", got ", packed.size(1));
  TORCH_CHECK(scales.numel() == R, "scales must be [R]=[", R, "], got ", scales.sizes());
  TORCH_CHECK(sums.size(0) == R && sums.size(1) == groups, "sums must be [R, K/128]=[", R, ", ", groups, "], got ",
              sums.sizes());
  TORCH_CHECK(words_a8 == 0 || R <= std::numeric_limits<int64_t>::max() / words_a8, "packed span exceeds int64");
  TORCH_CHECK(groups == 0 || R <= std::numeric_limits<int64_t>::max() / groups, "sums span exceeds int64");

  if (R > 0) {
    TORCH_CHECK(E > 0, "E=0 cannot cover R=", R);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 2 == 0, "packed must be 2-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0, "scales must be 4-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(sums.data_ptr()) % 4 == 0, "sums must be 4-byte aligned");
    TORCH_CHECK(!storage_overlaps(inputs, packed) && !storage_overlaps(inputs, scales) &&
                    !storage_overlaps(inputs, sums),
                "packed/scales/sums overlap inputs");
  }
  if (R == 0 || E == 0) return;

  const int device = inputs.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);

  const auto* inputs_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  auto* float_ptr = reinterpret_cast<half*>(float_output.data_ptr<at::Half>());
  auto* packed_ptr = packed.data_ptr<int16_t>();
  auto* scales_ptr = scales.data_ptr<float>();
  auto* sums_ptr = sums.data_ptr<int32_t>();
  const int* bits_ptr = activation_bits.data_ptr<int32_t>();
  const float* clips_ptr = clips.data_ptr<float>();
  const int rows = static_cast<int>(R);
  const int width = static_cast<int>(K);
  const int experts = static_cast<int>(E);
  const int stride_words = static_cast<int>(packed.size(1));

  dispatch_offset_type(offsets, [&](auto tag) {
    using OffsetT = typename decltype(tag)::type;
    const auto* offset_ptr = offsets.data_ptr<OffsetT>();
    if (silu) {
      launch_quantize<OffsetT, true>(inputs_ptr, offset_ptr, bits_ptr, clips_ptr, packed_ptr, scales_ptr, sums_ptr,
                                     float_ptr, rows, width, experts, stride_words, stream);
    } else {
      launch_quantize<OffsetT, false>(inputs_ptr, offset_ptr, bits_ptr, clips_ptr, packed_ptr, scales_ptr, sums_ptr,
                                      float_ptr, rows, width, experts, stride_words, stream);
    }
  });
}
