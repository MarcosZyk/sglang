#include "common.h"
#include "gemm.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

void decode_attention_cpu_tuned(
    at::Tensor& query,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap,
    int64_t head_block_num,
    int64_t num_kv_splits);

namespace {

#if defined(CPU_CAPABILITY_AVX512)
template <typename scalar_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst0,
    int ld_dst1,
    bool convert_v) {
  __m512i vinputs[16];
  int n = 0;
  for (; n < N; ++n) {
    vinputs[n] = _mm512_loadu_si512(src + ind[n] * ld_src);
  }
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  if (convert_v) {
    for (int n2 = 0; n2 < 16; n2 += 2) {
      __m512i d0, d1;
      std::tie(d0, d1) = transpose_2x32_16bit(vinputs[n2], vinputs[n2 + 1]);
      _mm512_storeu_si512(dst1 + (n2 >> 1) * ld_dst1 * 2, d0);
      _mm512_storeu_si512(dst1 + (n2 >> 1) * ld_dst1 * 2 + 32, d1);
    }
  }

  transpose_16x16_32bit(vinputs);
  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst0 + k * ld_dst0 * 2, vmask, vinputs[k]);
  }
}

template <typename scalar_t, typename index_t>
inline void pack_vnni_Kx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  int k = 0;
  for (; k < K; ++k) {
    vinputs[k] = _mm512_loadu_si512(src + ind[k] * ld_src);
  }
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}
#endif

template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst0,
    scalar_t* __restrict__ dst1,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int Kv,
    int ld_src,
    int ld_dst0,
    int ld_dst1) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int KBv = Kv / 32;

  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      int nb_size = std::min(N - nb * 16, 16);
      pack_vnni_Nx32<scalar_t, index_t>(
          dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2,
          dst1 + ((nb * 16) >> 1) * ld_dst1 * 2 + kb * 32 * 2,
          src + kb * 32,
          ind + nb * 16,
          nb_size,
          ld_src,
          ld_dst0,
          ld_dst1,
          kb < KBv);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = ind[n];
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst0[k * ld_dst0 * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
  for (int n = 0; n < (N >> 1) * 2; n += 2) {
    index_t index0 = ind[n + 0];
    index_t index1 = ind[n + 1];
    for (int k = 0; k < Kv; ++k) {
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 0] = src[index0 * ld_src + k];
      dst1[(n >> 1) * ld_dst1 * 2 + k * 2 + 1] = src[index1 * ld_src + k];
    }
  }
  if (N % 2 != 0) {
    index_t index = ind[N - 1];
    for (int k = 0; k < Kv; ++k) {
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 0] = src[index * ld_src + k];
      dst1[(N >> 1) * ld_dst1 * 2 + k * 2 + 1] = 0;
    }
  }
#endif
}

template <typename scalar_t, typename index_t>
void pack_vnni_value(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;

  for (int kb = 0; kb < KB; ++kb) {
    for (int nb = 0; nb < NB; ++nb) {
      int kb_size = std::min(K - kb * 2, 2);
      pack_vnni_Kx32<scalar_t, index_t>(
          dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          src + kb * 2 * ld_src + nb * 32,
          ind + kb * 2,
          kb_size,
          ld_src,
          ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = ind[k + 0];
    index_t index1 = ind[k + 1];
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    index_t index = ind[K - 1];
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
  }
#endif
}

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec out_bvec = bVec::loadu(src + d);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = src[d];
  }
}

template <typename scalar_t, int BLOCK_N>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

inline int64_t max_seq_len_in_batch(const int64_t* __restrict__ seq_lens, int64_t batches) {
  int64_t max_seq_len = 0;
  for (int64_t bs = 0; bs < batches; ++bs) {
    max_seq_len = std::max(max_seq_len, seq_lens[bs]);
  }
  return max_seq_len;
}

inline int64_t choose_decode_block_n(bool is_mla, int64_t head_size, int64_t max_seq_len) {
  UNUSED(max_seq_len);
  if (is_mla || head_size > 256) {
    return 128;
  }
  return 256;
}

inline bool use_grouped_packed_decode_kernel(
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t max_seq_len) {
  UNUSED(batches);
  UNUSED(max_seq_len);
  if (num_heads_kv <= 0 || num_heads % num_heads_kv != 0) {
    return false;
  }
  const int64_t num_groups = num_heads / num_heads_kv;
  return num_groups <= 8 && head_size <= 256 && head_size_v <= 256 &&
      head_size % TILE_K == 0 && head_size_v % TILE_K == 0;
}

template <typename scalar_t>
void decode_set_kv_buffer(
    scalar_t* __restrict__ k_buffer,
    scalar_t* __restrict__ v_buffer,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ loc,
    int64_t batches,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    int64_t nk_strideN,
    int64_t nk_strideH,
    int64_t nv_strideN,
    int64_t nv_strideH,
    bool is_mla) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; ++i) {
      int64_t loc_val = loc[bs];
      scalar_t* k_buffer_ptr = k_buffer + loc_val * k_strideN + head_kv_id * k_strideH;
      const scalar_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;
      copy_stub<scalar_t>(k_buffer_ptr, new_key_ptr, head_size);
      if (!is_mla) {
        scalar_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
        const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
        copy_stub<scalar_t>(v_buffer_ptr, new_value_ptr, head_size_v);
      }
      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

template <typename scalar_t>
void decode_accumulate_kv_splits(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t l_stride1,
    int64_t l_stride2) {
  using Vec = at::vec::Vectorized<float>;
  at::parallel_for(0, batches * num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      float* __restrict__ acc = attn_logits + i * l_stride1;
      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<scalar_t>::infinity();

      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * l_stride2;
        const float tlogic = (acc + kv_id * l_stride2)[head_size_v];
        float m_i = std::max(tlogic, m_prime);
        float m_delta = std::exp(m_prime - m_i);
        float e_logic = std::exp(tlogic - m_i);
        if (kv_id != 0) {
          at::vec::map2<float>(
              [m_delta, e_logic](Vec x, Vec y) { return x * Vec(m_delta) + y * Vec(e_logic); },
              acc,
              acc,
              tv,
              head_size_v);
        }
        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }
      copy_stub<scalar_t>(output + i * head_size_v, acc, 1.f / s_prime, head_size_v);
    }
  });
}

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_mla_tdm_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    scalar_t* __restrict__ buffer,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t num_blocks,
    int64_t block_size_h,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    int64_t buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;
  UNUSED(max_total_num_tokens);
  TORCH_CHECK(logit_cap == 0.f, "decode MLA: expect no logit_cap.");

  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;
  const int64_t num_worker_slots = num_blocks * num_kv_splits;

  at::parallel_for(0, num_worker_slots, 0, [&](int64_t begin, int64_t end) {
    const int tid = at::get_thread_num();
    scalar_t* __restrict__ Btmp0 = buffer + tid * buffer_size_per_thread;
    scalar_t* __restrict__ Btmp1 = Btmp0 + BLOCK_N * head_size;
    fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);

    alignas(64) float s_i[block_size_h * BLOCK_N];
    float* __restrict__ s_delta = s_i;
    alignas(64) scalar_t s_delta2[block_size_h * BLOCK_N];
    alignas(64) float s_prime[block_size_h];
    alignas(64) float m_prime[block_size_h];
    alignas(64) float m_delta[block_size_h];

    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t block_id = slot / num_kv_splits;
      const int64_t kv_id = slot % num_kv_splits;
      for (int64_t bs = 0; bs < batches; ++bs) {
        const int64_t h_start = block_id * block_size_h;
        const int64_t h_end = std::min(h_start + block_size_h, num_heads);
        const int64_t h_size = h_end - h_start;
        if (h_size <= 0) {
          continue;
        }

        const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + h_start * q_strideH;
        const int64_t seq_len_kv = seq_lens[bs];
        const int64_t req_pool_id = req_pool_indices[bs];
        TORCH_CHECK(seq_len_kv <= max_context_len, "seq_len_kv out of scope!");
        TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

        const int64_t split_size = div_up(seq_len_kv, num_kv_splits);
        const int64_t kv_start = kv_id * split_size;
        const int64_t kv_end = std::min(kv_start + split_size, seq_len_kv);

        fill_stub(s_prime, 0.f, block_size_h);
        fill_stub(m_prime, -std::numeric_limits<float>::infinity(), block_size_h);

        float* __restrict__ v_prime = attn_logits + bs * l_stride0 + h_start * l_stride1 + kv_id * l_stride2;
        for (int64_t h = 0; h < h_size; ++h) {
          fill_stub(v_prime + h * l_stride1, 0.f, head_size_v);
        }

        for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
          const int64_t n_size = std::min(BLOCK_N, kv_end - n);
          const int64_t padded_n_size = div_up(int(n_size), TILE_K) * TILE_K;
          const index_t* __restrict__ indices = req_to_token + req_pool_id * max_context_len + n;

          pack_vnni<scalar_t, index_t>(
              Btmp0,
              Btmp1,
              k_buffer + 0 * k_strideH,
              indices,
              n_size,
              head_size,
              head_size_v,
              k_strideN,
              BLOCK_N,
              head_size_v);

          at::native::cpublas::brgemm(
              h_size,
              n_size,
              head_size,
              q_strideH,
              BLOCK_N,
              BLOCK_N,
              false,
              q_ptr,
              Btmp0,
              s_i);

          const Vec scale_vec = Vec(scaling);
          for (int64_t h = 0; h < h_size; ++h) {
            at::vec::map<float>(
                [scale_vec](Vec x) { return x * scale_vec; }, s_i + h * BLOCK_N, s_i + h * BLOCK_N, n_size);

            float m_i = at::vec::reduce_all<float>(
                [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + h * BLOCK_N, n_size);
            m_i = std::max(m_i, m_prime[h]);
            m_delta[h] = std::exp(m_prime[h] - m_i);

            at::vec::map<float>(
                [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); },
                s_delta + h * BLOCK_N,
                s_i + h * BLOCK_N,
                n_size);

            s_prime[h] *= m_delta[h];
            s_prime[h] += at::vec::reduce_all<float>(
                [](Vec& x, Vec& y) { return x + y; }, s_delta + h * BLOCK_N, n_size);
            m_prime[h] = m_i;

            const float scale_m = m_delta[h];
            at::vec::map<float>(
                [scale_m](Vec x) { return x * Vec(scale_m); },
                v_prime + h * l_stride1,
                v_prime + h * l_stride1,
                head_size_v);

            fill_stub(s_delta + h * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
            copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
          }

          at::native::cpublas::brgemm(
              h_size,
              head_size_v,
              padded_n_size,
              BLOCK_N,
              head_size_v,
              l_stride1,
              true,
              s_delta2,
              Btmp1,
              v_prime);
        }

        if (kv_end > kv_start) {
          for (int64_t h = 0; h < h_size; ++h) {
            const float s = 1.f / s_prime[h];
            at::vec::map<float>(
                [s](Vec out) { return out * Vec(s); }, v_prime + h * l_stride1, v_prime + h * l_stride1, head_size_v);
            (v_prime + h * l_stride1)[head_size_v] = m_prime[h] + std::log(s_prime[h]);
          }
        }
      }
    }
    at::native::cpublas::brgemm_release();
  });

  decode_accumulate_kv_splits(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

template <typename scalar_t, typename index_t, int64_t BLOCK_N>
void decode_attention_grouped_packed_tdm_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    scalar_t* __restrict__ buffer,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t num_blocks,
    int64_t block_size_h,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    int64_t buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;
  UNUSED(max_total_num_tokens);

  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;
  const bool has_logit_cap = logit_cap > 0;
  const float rlogit_cap = has_logit_cap ? 1.f / static_cast<float>(logit_cap) : 0.f;
  const int64_t num_groups = num_heads / num_heads_kv;
  const int64_t num_worker_slots = num_blocks * num_kv_splits;

  at::parallel_for(0, num_worker_slots, 0, [&](int64_t begin, int64_t end) {
    const int tid = at::get_thread_num();
    scalar_t* __restrict__ Btmp0 = buffer + tid * buffer_size_per_thread;
    scalar_t* __restrict__ Btmp1 = Btmp0 + BLOCK_N * head_size;
    fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);

    alignas(64) float s_i[block_size_h * BLOCK_N];
    float* __restrict__ s_delta = s_i;
    alignas(64) scalar_t s_delta2[block_size_h * BLOCK_N];
    alignas(64) float s_prime[block_size_h];
    alignas(64) float m_prime[block_size_h];
    alignas(64) float m_delta[block_size_h];

    for (int64_t slot = begin; slot < end; ++slot) {
      const int64_t block_id = slot / num_kv_splits;
      const int64_t kv_id = slot % num_kv_splits;
      for (int64_t bs = 0; bs < batches; ++bs) {
        const int64_t seq_len_kv = seq_lens[bs];
        const int64_t req_pool_id = req_pool_indices[bs];
        TORCH_CHECK(seq_len_kv <= max_context_len, "seq_len_kv out of scope!");
        TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

        const int64_t split_size = div_up(seq_len_kv, num_kv_splits);
        const int64_t kv_start = kv_id * split_size;
        const int64_t kv_end = std::min(kv_start + split_size, seq_len_kv);

        for (int64_t head_kv_id = 0; head_kv_id < num_heads_kv; ++head_kv_id) {
          const int64_t h_start = head_kv_id * num_groups + block_id * block_size_h;
          const int64_t h_end = std::min(h_start + block_size_h, head_kv_id * num_groups + num_groups);
          const int64_t h_size = h_end - h_start;
          if (h_size <= 0) {
            continue;
          }

          const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + h_start * q_strideH;
          fill_stub(s_prime, 0.f, block_size_h);
          fill_stub(m_prime, -std::numeric_limits<float>::infinity(), block_size_h);

          float* __restrict__ v_prime = attn_logits + bs * l_stride0 + h_start * l_stride1 + kv_id * l_stride2;
          for (int64_t h = 0; h < h_size; ++h) {
            fill_stub(v_prime + h * l_stride1, 0.f, head_size_v);
          }

          for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
            const int64_t n_size = std::min(BLOCK_N, kv_end - n);
            const int64_t padded_n_size = div_up(int(n_size), TILE_K) * TILE_K;
            const index_t* __restrict__ indices = req_to_token + req_pool_id * max_context_len + n;

            pack_vnni<scalar_t, index_t>(
                Btmp0,
                Btmp1,
                k_buffer + head_kv_id * k_strideH,
                indices,
                n_size,
                head_size,
                0,
                k_strideN,
                BLOCK_N,
                head_size_v);

            at::native::cpublas::brgemm(
                h_size,
                n_size,
                head_size,
                q_strideH,
                BLOCK_N,
                BLOCK_N,
                false,
                q_ptr,
                Btmp0,
                s_i);

            const Vec scale_vec = Vec(scaling);
            for (int64_t h = 0; h < h_size; ++h) {
              float* __restrict__ s_row = s_i + h * BLOCK_N;
              if (has_logit_cap) {
                at::vec::map<float>(
                    [scale_vec, logit_cap, rlogit_cap](Vec x) {
                      return apply_logit_cap(x * scale_vec, static_cast<float>(logit_cap), rlogit_cap);
                    },
                    s_row,
                    s_row,
                    n_size);
              } else {
                at::vec::map<float>(
                    [scale_vec](Vec x) { return x * scale_vec; }, s_row, s_row, n_size);
              }

              float m_i = at::vec::reduce_all<float>(
                  [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_row, n_size);
              m_i = std::max(m_i, m_prime[h]);
              m_delta[h] = std::exp(m_prime[h] - m_i);

              at::vec::map<float>(
                  [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); },
                  s_delta + h * BLOCK_N,
                  s_row,
                  n_size);

              s_prime[h] *= m_delta[h];
              s_prime[h] += at::vec::reduce_all<float>(
                  [](Vec& x, Vec& y) { return x + y; }, s_delta + h * BLOCK_N, n_size);
              m_prime[h] = m_i;

              const float scale_m = m_delta[h];
              at::vec::map<float>(
                  [scale_m](Vec x) { return x * Vec(scale_m); },
                  v_prime + h * l_stride1,
                  v_prime + h * l_stride1,
                  head_size_v);

              fill_stub(s_delta + h * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
              copy_stub<scalar_t, BLOCK_N>(s_delta2 + h * BLOCK_N, s_delta + h * BLOCK_N);
            }

            pack_vnni_value<scalar_t, index_t>(
                Btmp1,
                v_buffer + head_kv_id * v_strideH,
                indices,
                n_size,
                head_size_v,
                v_strideN,
                head_size_v);

            at::native::cpublas::brgemm(
                h_size,
                head_size_v,
                padded_n_size,
                BLOCK_N,
                head_size_v,
                l_stride1,
                true,
                s_delta2,
                Btmp1,
                v_prime);
          }

          if (kv_end > kv_start) {
            for (int64_t h = 0; h < h_size; ++h) {
              const float s = 1.f / s_prime[h];
              at::vec::map<float>(
                  [s](Vec out) { return out * Vec(s); }, v_prime + h * l_stride1, v_prime + h * l_stride1, head_size_v);
              (v_prime + h * l_stride1)[head_size_v] = m_prime[h] + std::log(s_prime[h]);
            }
          }
        }
      }  // slot
    }
    at::native::cpublas::brgemm_release();
  });

  decode_accumulate_kv_splits(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

}  // namespace

void decode_attention_cpu_tdm(
    at::Tensor& query,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap,
    int64_t head_block_num) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_cpu_tdm",
      std::vector<c10::IValue>(
          {query, output, k_cache, v_cache, attn_logits, req_to_token, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_DIM(3, query);
  CHECK_DIM(3, k_cache);
  CHECK_DIM(3, v_cache);
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);
  CHECK_DIM(1, loc);

  TORCH_CHECK(head_block_num > 0, "decode_tdm: expect head_block_num > 0, got ", head_block_num);

  const int64_t num_threads = at::get_num_threads();
  TORCH_CHECK(num_threads > 0, "decode_tdm: expect num_threads > 0, got ", num_threads);
  TORCH_CHECK(
      num_threads % head_block_num == 0,
      "decode_tdm: expect num_threads % head_block_num == 0, got num_threads=",
      num_threads,
      ", head_block_num=",
      head_block_num);

  const int64_t num_kv_splits = num_threads / head_block_num;
  TORCH_CHECK(num_kv_splits > 0, "decode_tdm: derived num_kv_splits must be > 0, got ", num_kv_splits);

  const int64_t num_seqs = seq_lens.size(0);
  const int64_t max_num_reqs = req_to_token.size(0);
  const int64_t max_context_len = req_to_token.size(1);
  const int64_t max_total_num_tokens = k_cache.size(0);

  const int64_t num_heads = query.size(1);
  const int64_t num_heads_kv = k_cache.size(1);
  const int64_t head_size = query.size(2);
  const int64_t head_size_v = v_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0, "decode_tdm: expect num_heads_kv > 0, got ", num_heads_kv);
  TORCH_CHECK(
      num_heads % num_heads_kv == 0,
      "decode_tdm: expect num_heads divisible by num_heads_kv, got ",
      num_heads,
      " and ",
      num_heads_kv);

  CHECK_EQ(loc.numel(), num_seqs);
  CHECK_EQ(attn_logits.size(0), num_seqs);
  CHECK_EQ(attn_logits.size(1), num_heads);
  CHECK_EQ(attn_logits.size(2), num_kv_splits);
  CHECK_EQ(attn_logits.size(3), head_size_v + 1);
  CHECK_EQ(attn_logits.scalar_type(), at::kFloat);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode_tdm: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "decode_tdm: expect seq_lens to be int64.");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong, "decode_tdm: expect req_pool_indices to be int64.");

  void* k_buffer_data = k_cache.data_ptr();
  void* v_buffer_data = v_cache.data_ptr();
  const bool is_mla = (k_buffer_data == v_buffer_data) && (num_heads_kv == 1) && (head_size == head_size_v + 64);
  const int64_t max_seq_len = max_seq_len_in_batch(seq_lens.data_ptr<int64_t>(), num_seqs);
  const int64_t block_n = choose_decode_block_n(is_mla, head_size, max_seq_len);
  const bool use_grouped_packed = !is_mla &&
      (num_heads != num_heads_kv) &&
      use_grouped_packed_decode_kernel(num_seqs, num_heads, num_heads_kv, head_size, head_size_v, max_seq_len);

  if (!is_mla && !use_grouped_packed) {
    decode_attention_cpu_tuned(
        query,
        k_cache,
        v_cache,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
        head_block_num,
        num_kv_splits);
    return;
  }

  constexpr int64_t BLOCK_N_MLA = 128;
  constexpr int64_t BLOCK_N_GQA_PACKED = 128;
  if (is_mla) {
    TORCH_CHECK(block_n == BLOCK_N_MLA, "decode_tdm MLA: unsupported BLOCK_N ", block_n);
  }

  const int64_t q_strideM = query.stride(0);
  const int64_t q_strideH = query.stride(1);
  const int64_t k_strideN = k_cache.stride(0);
  const int64_t k_strideH = k_cache.stride(1);
  const int64_t v_strideN = v_cache.stride(0);
  const int64_t v_strideH = v_cache.stride(1);
  const int64_t nk_strideN = key.stride(0);
  const int64_t nk_strideH = key.stride(1);
  const int64_t nv_strideN = value.stride(0);
  const int64_t nv_strideH = value.stride(1);

  const int64_t num_blocks = head_block_num;
  int64_t block_size_h = 0;
  if (is_mla) {
    block_size_h = div_up(num_heads, num_blocks);
  } else {
    const int64_t num_groups = num_heads / num_heads_kv;
    block_size_h = div_up(num_groups, num_blocks);
  }
  const int64_t size_per_thread = (is_mla ? BLOCK_N_MLA : BLOCK_N_GQA_PACKED) * (head_size + head_size_v);
  auto buffer = at::empty({num_threads, size_per_thread}, k_cache.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_tdm_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_tdm_indices", [&] {
      decode_set_kv_buffer(
          (scalar_t*)k_buffer_data,
          (scalar_t*)v_buffer_data,
          key.data_ptr<scalar_t>(),
          value.data_ptr<scalar_t>(),
          loc.data_ptr<int64_t>(),
          num_seqs,
          num_heads_kv,
          head_size,
          head_size_v,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          nk_strideN,
          nk_strideH,
          nv_strideN,
          nv_strideH,
          is_mla);

      if (is_mla) {
        decode_attention_mla_tdm_kernel_impl<scalar_t, index_t, BLOCK_N_MLA>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const scalar_t*)k_buffer_data,
            (const scalar_t*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            buffer.data_ptr<scalar_t>(),
            num_seqs,
            num_heads,
            head_size,
            head_size_v,
            num_kv_splits,
            num_blocks,
            block_size_h,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            size_per_thread);
      } else {
        decode_attention_grouped_packed_tdm_kernel_impl<scalar_t, index_t, BLOCK_N_GQA_PACKED>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const scalar_t*)k_buffer_data,
            (const scalar_t*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            buffer.data_ptr<scalar_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            num_blocks,
            block_size_h,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            size_per_thread);
      }
    });
  });
}
