#include "common.h"
#include "gemm.h"
#include "vec.h"

#include<iostream>

namespace {

// [NOTE]: extend attention for CPU
//   1. tune BLOCK_M and BLOCK_N
//   2. can handle non-contiguous k_exttend and v_extend
//   3. computes attention for prefix and extend separately
//   4. BLOCK_M=16 tuned for EAGLE speculative decoding (draft_tokens=16-20)
//   5. BLOCK_N=64 tuned for long sequences (8-16k context)
//

template <typename index_t>
inline index_t get_index(index_t* ind, int i) {
  return (ind == nullptr) ? (index_t)i : ind[i];
}

#if defined(CPU_CAPABILITY_AVX512)
// key: from [N, 32] to [32/2, N, 2]
template <typename scalar_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    vinputs[n] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask, vinputs[k]);
  }
}

// value: from [K, 32] to [K/2, 32, 2]
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
    index_t index = get_index(ind, k);
    vinputs[k] = _mm512_loadu_si512(src + index * ld_src);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}
#endif

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;  // no remainder
  const bool is_indexed = ind != nullptr;

  for (int nb = 0; nb < NB; ++nb) {
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      int nb_size = std::min(N - nb * 16, 16);
      pack_vnni_Nx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + kb * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = get_index(ind, n);
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
#endif
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;  // no remainder
  const bool is_indexed = ind != nullptr;

  for (int kb = 0; kb < KB; ++kb) {
    for (int nb = 0; nb < NB; ++nb) {
      // handle 2x512bits each block
      int kb_size = std::min(K - kb * 2, 2);
      pack_vnni_Kx32<scalar_t, index_t>(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + nb * 32,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = get_index(ind, k + 0);
    index_t index1 = get_index(ind, k + 1);
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    index_t index = get_index(ind, K - 1);
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
#endif
}

template <typename scalar_t, typename index_t>
void pack_vnni_mla(
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
      __m512i vinputs[16];
      int n = 0;
      for (; n < nb_size; ++n) {
        index_t index = ind[nb * 16 + n];
        vinputs[n] = _mm512_loadu_si512(src + index * ld_src + kb * 32);
      }
      for (; n < 16; ++n) {
        vinputs[n] = _mm512_set1_epi32(0);
      }

      if (kb < KBv) {
        for (int nn = 0; nn < 16; nn += 2) {
          __m512i d0, d1;
          std::tie(d0, d1) = transpose_2x32_16bit(vinputs[nn], vinputs[nn + 1]);
          _mm512_storeu_si512(dst1 + ((nb * 16 + nn) >> 1) * ld_dst1 * 2 + kb * 32 * 2, d0);
          _mm512_storeu_si512(dst1 + ((nb * 16 + nn) >> 1) * ld_dst1 * 2 + kb * 32 * 2 + 32, d1);
        }
      }

      transpose_16x16_32bit(vinputs);
      const __mmask16 vmask = (1 << nb_size) - 1;
      for (int k = 0; k < 16; ++k) {
        _mm512_mask_storeu_epi32(
            dst0 + ((kb * 32) >> 1) * ld_dst0 * 2 + nb * 16 * 2 + k * ld_dst0 * 2,
            vmask,
            vinputs[k]);
      }
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

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
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
    // for COLS = 2, 4 use 512bit store
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_fvec = fVec(s);
  int d = 0;
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

template <typename value_t>
inline int64_t max_value_in_batch(const value_t* __restrict__ data, int batches) {
  int64_t max_value = 0;
  for (int bs = 0; bs < batches; ++bs) {
    max_value = std::max<int64_t>(max_value, static_cast<int64_t>(data[bs]));
  }
  return max_value;
}

inline int choose_extend_block_n_impl(int64_t max_kv_len, int num_heads_kv, int head_size) {
  // Long-prefix GQA is bandwidth dominated, so a wider KV tile reduces loop / pack overhead.
  // MLA (single KV head, larger K) keeps BLOCK_N=64 to avoid blowing up the per-thread AMX
  // packing buffer and to preserve L2 locality.
  if (num_heads_kv > 1 && head_size <= 256 && max_kv_len >= 4096) {
    return 128;
  }
  return 64;
}

template <typename index_t>
inline int choose_extend_block_n(
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    int batches,
    int num_heads_kv,
    int head_size,
    bool is_prefix_skipped) {
  UNUSED(extend_seq_lens);
  if (is_prefix_skipped) {
    return 64;
  }
  return choose_extend_block_n_impl(
      max_value_in_batch(seq_lens, batches), num_heads_kv, head_size);
}

template <typename index_t>
inline int choose_treemask_block_n(
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    int batches,
    int num_heads_kv,
    int head_size) {
  int64_t max_total_kv_len = 0;
  for (int bs = 0; bs < batches; ++bs) {
    max_total_kv_len = std::max<int64_t>(
        max_total_kv_len, seq_lens[bs] + static_cast<int64_t>(extend_seq_lens[bs]));
  }
  return choose_extend_block_n_impl(max_total_kv_len, num_heads_kv, head_size);
}

inline bool is_mla_shared_kv_layout(const at::Tensor& k_buffer, const at::Tensor& v_buffer, int num_heads_kv) {
  return k_buffer.data_ptr() == v_buffer.data_ptr() && num_heads_kv == 1 &&
      k_buffer.size(2) == v_buffer.size(2) + 64;
}

template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int buffer_size_per_thread,
    bool is_prefix_skipped) {
  using Vec = at::vec::Vectorized<float>;

  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of blocks along M
  int MB = div_up(max_len_extend, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta2: [BLOCK_M, BLOCK_N]; copy of s_delta in scalar_t
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = s_delta2 + BLOCK_M * BLOCK_N;

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];
    alignas(64) float m_delta[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      // seq_len = prefix + extend
      int head_kv_id = head_id / num_groups;
      int seq_len = seq_lens[bs];
      int seq_len_extend = extend_seq_lens[bs];
      int seq_len_prefix = seq_len - seq_len_extend;
      int seq_extend_start_loc = extend_start_loc[bs];

      int req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      if (is_prefix_skipped) {
        TORCH_CHECK(seq_len_prefix == 0, "extend attention: expect seq_len_prefix to be 0, got ", seq_len_prefix);
      }

      // offset and size in MB
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

      // get query
      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<float>::infinity(), m_size);

      // stage 1: compute scores with prefix
      for (int n = 0; n < seq_len_prefix; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, seq_len_prefix - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_buffer + head_kv_id * k_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // fused scale + softmax update per row
        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          float* __restrict__ s_row = s_i + row * BLOCK_N;

          // fused: s_i <- s_i * scale, then optionally apply logit_cap
          if (has_logit_cap) {
            at::vec::map<float>(
                [scale_vec, logit_cap, rlogit_cap](Vec x) { return apply_logit_cap(x * scale_vec, logit_cap, rlogit_cap); },
                s_row, s_row, n_size);
          } else {
            at::vec::map<float>(
                [scale_vec](Vec x) { return x * scale_vec; }, s_row, s_row, n_size);
          }

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_row, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          m_delta[row] = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); }, s_delta + row * BLOCK_N, s_row, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta[row];
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          float scale_m = m_delta[row];
          at::vec::map<float>(
              [scale_m](Vec x) { return x * Vec(scale_m); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_buffer + head_kv_id * v_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_prefix

      // stage 2: compute the triangle part
      int num_keys = std::min(seq_len_extend, m + BLOCK_M);
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_extend + (seq_extend_start_loc + n) * ke_strideN + head_kv_id * ke_strideH,
            /*    ind */ nullptr,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ ke_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // apply causal mask
        if (num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            int last_col = m + row - n;
            // fill [last_col + 1, n_size) to -inf
            float* row_ptr = s_i + row * BLOCK_N;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
          }
        }

        // fused scale + softmax update per row
        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          float* __restrict__ s_row = s_i + row * BLOCK_N;

          // fused: s_i <- s_i * scale, then optionally apply logit_cap
          if (has_logit_cap) {
            at::vec::map<float>(
                [scale_vec, logit_cap, rlogit_cap](Vec x) { return apply_logit_cap(x * scale_vec, logit_cap, rlogit_cap); },
                s_row, s_row, n_size);
          } else {
            at::vec::map<float>(
                [scale_vec](Vec x) { return x * scale_vec; }, s_row, s_row, n_size);
          }

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_row, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          m_delta[row] = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); }, s_delta + row * BLOCK_N, s_row, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta[row];
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          float scale_m = m_delta[row];
          at::vec::map<float>(
              [scale_m](Vec x) { return x * Vec(scale_m); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_extend + (seq_extend_start_loc + n) * ve_strideN + head_kv_id * ve_strideH,
            /*    ind */ nullptr,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ ve_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_extend

      scalar_t* __restrict__ out_ptr = o_extend + (seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

// Tree-mask variant of extend attention.
//
// When a custom_mask (tree mask) is provided, all KV tokens are read from
// k_buffer/v_buffer via req_to_token (prefix + extend are already stored
// there by save_kv_cache).  The tree mask replaces the causal mask to
// control which KV positions each query token can attend to.
//
// custom_mask layout: 1-D bool tensor, packed per-batch as
//   [draft_len_0 * total_kv_len_0, draft_len_1 * total_kv_len_1, ...]
// where total_kv_len_i = seq_lens[i] + extend_seq_lens[i] (prefix + draft)
// and draft_len_i = extend_seq_lens[i].
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_treemask_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const bool* __restrict__ custom_mask,
    const int64_t* __restrict__ mask_offsets,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    bool is_mla_shared_kv,
    int buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;

  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  int MB = div_up(max_len_extend, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);
    scalar_t* __restrict__ Btmp0 = s_delta2 + BLOCK_M * BLOCK_N;
    scalar_t* __restrict__ Btmp1 = Btmp0 + (is_mla_shared_kv ? BLOCK_N * head_size : 0);

    if (is_mla_shared_kv) {
      fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);
    } else {
      fill_stub(Btmp0, 0.f, BLOCK_N * std::max(head_size, head_size_v));
    }

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];
    alignas(64) float m_delta[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int head_kv_id = head_id / num_groups;
      int seq_len_extend = extend_seq_lens[bs];
      int seq_extend_start_loc = extend_start_loc[bs];
      // In EAGLE verify, seq_lens is the prefix length (tokens already in KV cache).
      // The total KV length that the mask indexes is prefix + extend.
      int total_kv_len = static_cast<int>(seq_lens[bs]) + seq_len_extend;

      int req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(total_kv_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

      // The mask for this batch starts at mask_offsets[bs].
      // mask_2d[row][col] = custom_mask[mask_base + row * total_kv_len + col]
      const int64_t mask_base = mask_offsets[bs];

      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      // Single unified loop over all KV tokens (prefix + extend),
      // reading from k_buffer/v_buffer via req_to_token.
      for (int n = 0; n < total_kv_len; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, total_kv_len - n);
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        const index_t* __restrict__ indices = req_to_token + req_pool_id * max_context_len + n;

        if (is_mla_shared_kv) {
          pack_vnni_mla<scalar_t, index_t>(
              /*    dst0 */ Btmp0,
              /*    dst1 */ Btmp1,
              /*     src */ k_buffer + head_kv_id * k_strideH,
              /*     ind */ indices,
              /*       N */ n_size,
              /*       K */ head_size,
              /*      Kv */ head_size_v,
              /*  ld_src */ k_strideN,
              /* ld_dst0 */ BLOCK_N,
              /* ld_dst1 */ head_size_v);
        } else {
          pack_vnni<scalar_t, index_t>(
              /*    dst */ Btmp0,
              /*    src */ k_buffer + head_kv_id * k_strideH,
              /*    ind */ indices,
              /*     N  */ n_size,
              /*     K  */ head_size,
              /* ld_src */ k_strideN,
              /* ld_dst */ BLOCK_N);
        }

        // s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp0,
            /* C     */ s_i);

        // fused scale + mask + softmax update per row
        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          float* __restrict__ s_row = s_i + row * BLOCK_N;

          // fused: s_i <- s_i * scale, then optionally apply logit_cap
          if (has_logit_cap) {
            at::vec::map<float>(
                [scale_vec, logit_cap, rlogit_cap](Vec x) { return apply_logit_cap(x * scale_vec, logit_cap, rlogit_cap); },
                s_row, s_row, n_size);
          } else {
            at::vec::map<float>(
                [scale_vec](Vec x) { return x * scale_vec; }, s_row, s_row, n_size);
          }

          // Apply tree mask: set masked positions to -inf
          // Vectorized with AVX512 for better throughput
          const bool* mask_row = custom_mask + mask_base + (m + row) * total_kv_len + n;
#if defined(CPU_CAPABILITY_AVX512)
          {
            const Vec neg_inf_vec = Vec(-std::numeric_limits<float>::infinity());
            int col = 0;
            // Process 16 elements at a time using AVX512 mask operations
            for (; col <= n_size - 16; col += 16) {
              // Load 16 bool values and create a bitmask
              __m128i mask_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask_row + col));
              __mmask16 kmask = _mm_test_epi8_mask(mask_bytes, _mm_set1_epi8(1));
              // Blend: keep original where mask=true, set -inf where mask=false
              __m512 vals = _mm512_loadu_ps(s_row + col);
              vals = _mm512_mask_blend_ps(kmask, (__m512)neg_inf_vec, vals);
              _mm512_storeu_ps(s_row + col, vals);
            }
            // Handle remaining elements
            for (; col < n_size; ++col) {
              if (!mask_row[col]) {
                s_row[col] = -std::numeric_limits<float>::infinity();
              }
            }
          }
#else   // CPU_CAPABILITY_AVX512
          int col = 0;
          // Process 4 elements at a time for better ILP
          for (; col <= n_size - 4; col += 4) {
            if (!mask_row[col + 0]) s_row[col + 0] = -std::numeric_limits<float>::infinity();
            if (!mask_row[col + 1]) s_row[col + 1] = -std::numeric_limits<float>::infinity();
            if (!mask_row[col + 2]) s_row[col + 2] = -std::numeric_limits<float>::infinity();
            if (!mask_row[col + 3]) s_row[col + 3] = -std::numeric_limits<float>::infinity();
          }
          // Handle remaining elements
          for (; col < n_size; ++col) {
            if (!mask_row[col]) {
              s_row[col] = -std::numeric_limits<float>::infinity();
            }
          }
#endif

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_row, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          m_delta[row] = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); }, s_delta + row * BLOCK_N, s_row, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta[row];
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          float scale_m = m_delta[row];
          at::vec::map<float>(
              [scale_m](Vec x) { return x * Vec(scale_m); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        if (!is_mla_shared_kv) {
          pack_vnni2<scalar_t, index_t>(
              /*    dst */ Btmp0,
              /*    src */ v_buffer + head_kv_id * v_strideH,
              /*    ind */ indices,
              /*     K  */ n_size,
              /*     N  */ head_size_v,
              /* ld_src */ v_strideN,
              /* ld_dst */ head_size_v);
        }

        // V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ is_mla_shared_kv ? Btmp1 : Btmp0,
            /* C     */ v_prime);
      }  // loop over all KV tokens

      scalar_t* __restrict__ out_ptr = o_extend + (seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

// Accumulate partial attention results from KV splits for extend attention.
//
// Each split stores BLOCK_M rows of [head_size_v values, m+log(s) header]
// in attn_logits with layout:
//   attn_logits[work_id][kv_id][row * (head_size_v + 1) + ...]
//
// This merges them using the log-sum-exp trick (same as decode_accumulate_kv_splits)
// but extended to handle BLOCK_M rows per work item.
//
template <typename scalar_t, int BLOCK_M>
void extend_accumulate_kv_splits(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const int* __restrict__ work_m_sizes,
    int total_work_items,
    int num_kv_splits,
    int head_size_v,
    int o_strideM,
    int o_strideH,
    const int* __restrict__ work_seq_start,
    const int* __restrict__ work_head_ids) {
  using Vec = at::vec::Vectorized<float>;

  // stride per row within a split: head_size_v values + 1 header
  const int row_stride = head_size_v + 1;
  // stride per split
  const int split_stride = BLOCK_M * row_stride;
  // stride per work item (all splits)
  const int work_stride = num_kv_splits * split_stride;

  // parallel on [total_work_items * BLOCK_M] for fine-grained parallelism
  at::parallel_for(0, total_work_items, 0, [&](int begin, int end) {
    for (int wi = begin; wi < end; ++wi) {
      int m_size = work_m_sizes[wi];
      float* __restrict__ base = attn_logits + wi * work_stride;

      for (int row = 0; row < m_size; ++row) {
        // acc points to the first split's data for this row
        float* __restrict__ acc = base + row * row_stride;

        float s_prime = 0.f;
        // Use scalar_t infinity (e.g. bf16 max ~65504) instead of float
        // infinity to avoid NaN when the first split is empty:
        //   max(-inf, -inf) = -inf  =>  exp(-inf - (-inf)) = exp(NaN) = NaN
        // With -65504:
        //   max(-inf, -65504) = -65504  =>  exp(-inf - (-65504)) = 0  (correct)
        float m_prime = -std::numeric_limits<scalar_t>::infinity();

        for (int kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
          float* __restrict__ tv = base + kv_id * split_stride + row * row_stride;
          const float tlogic = tv[head_size_v];

          float m_i = std::max(tlogic, m_prime);
          float m_delta_val = std::exp(m_prime - m_i);
          float e_logic = std::exp(tlogic - m_i);

          if (kv_id != 0) {
            at::vec::map2<float>(
                [m_delta_val, e_logic](Vec x, Vec y) {
                  return x * Vec(m_delta_val) + y * Vec(e_logic);
                },
                acc, acc, tv, head_size_v);
          }

          s_prime = s_prime * m_delta_val + e_logic;
          m_prime = m_i;
        }

        // Write final output
        int seq_offset = work_seq_start[wi];
        int head_id = work_head_ids[wi];
        scalar_t* __restrict__ out_ptr = output + seq_offset * o_strideM + head_id * o_strideH + row * o_strideM;
        copy_stub<scalar_t>(out_ptr, acc, 1.f / s_prime, head_size_v);
      }
    }
  });
}

// Sequence-parallel variant of extend_attention_treemask_kernel_impl.
//
// Splits the KV sequence dimension across multiple threads so that
// when batches * num_heads * MB is small (e.g., 8), we can still
// utilize all CPU cores by adding a num_kv_splits dimension.
//
// Parallel on [batches, num_heads, MB, num_kv_splits].
// Each (bs, head, mb, kv_id) computes partial attention over its KV range,
// storing v_prime and m+log(s) into attn_logits.
// After the parallel region, extend_accumulate_kv_splits merges the splits.
//
template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_treemask_sp_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const bool* __restrict__ custom_mask,
    const int64_t* __restrict__ mask_offsets,
    float* __restrict__ attn_logits,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int num_kv_splits,
    bool is_mla_shared_kv,
    int buffer_size_per_thread) {
  using Vec = at::vec::Vectorized<float>;

  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;
  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  int MB = div_up(max_len_extend, BLOCK_M);

  // attn_logits layout per work item (bs, head, mb):
  //   [num_kv_splits, BLOCK_M * (head_size_v + 1)]
  // The +1 stores m + log(s) for the log-sum-exp merge.
  const int row_stride = head_size_v + 1;
  const int split_stride = BLOCK_M * row_stride;
  const int work_stride = num_kv_splits * split_stride;

  // We also need work_m_sizes, work_seq_start, work_head_ids for accumulate.
  // These are computed during the parallel loop and stored in arrays.
  const int total_work_items = batches * num_heads * MB;

  // Temporary arrays for accumulate metadata
  // Allocated on stack or heap depending on size
  std::vector<int> work_m_sizes(total_work_items, 0);
  std::vector<int> work_seq_start(total_work_items, 0);
  std::vector<int> work_head_ids(total_work_items, 0);

  // Pre-compute work metadata
  {
    int bs = 0, head_id = 0, mb = 0;
    for (int wi = 0; wi < total_work_items; ++wi) {
      if (wi == 0) {
        data_index_init(0, bs, batches, head_id, num_heads, mb, MB);
      }
      int seq_len_extend = extend_seq_lens[bs];
      int seq_extend_start_loc_val = extend_start_loc[bs];
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      work_m_sizes[wi] = std::max(m_size, 0);
      work_seq_start[wi] = seq_extend_start_loc_val + m;
      work_head_ids[wi] = head_id;

      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
  }

  // parallel on [batches, num_heads, MB, num_kv_splits]
  at::parallel_for(0, total_work_items * num_kv_splits, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB, kv_id, num_kv_splits);

    int tid = at::get_thread_num();
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;
    float* __restrict__ v_prime_local = s_i + BLOCK_M * BLOCK_N;
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime_local + BLOCK_M * head_size_v);
    scalar_t* __restrict__ Btmp0 = s_delta2 + BLOCK_M * BLOCK_N;
    scalar_t* __restrict__ Btmp1 = Btmp0 + (is_mla_shared_kv ? BLOCK_N * head_size : 0);

    if (is_mla_shared_kv) {
      fill_stub(Btmp1, 0.f, BLOCK_N * head_size_v);
    } else {
      fill_stub(Btmp0, 0.f, BLOCK_N * std::max(head_size, head_size_v));
    }

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];
    alignas(64) float m_delta[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      int head_kv_id = head_id / num_groups;
      int seq_len_extend = extend_seq_lens[bs];
      int seq_extend_start_loc_val = extend_start_loc[bs];
      int total_kv_len = static_cast<int>(seq_lens[bs]) + seq_len_extend;

      int req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(total_kv_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        // Mark this split as empty: zero data + header = -inf
        int work_id = bs * num_heads * MB + head_id * MB + mb;
        float* __restrict__ v_prime_out = attn_logits + work_id * work_stride + kv_id * split_stride;
        for (int row = 0; row < BLOCK_M; ++row) {
          fill_stub(v_prime_out + row * row_stride, 0.f, head_size_v);
          v_prime_out[row * row_stride + head_size_v] = -std::numeric_limits<float>::infinity();
        }
        data_index_step(bs, batches, head_id, num_heads, mb, MB, kv_id, num_kv_splits);
        continue;
      }

      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc_val + m) * q_strideM + head_id * q_strideH;

      const int64_t mask_base = mask_offsets[bs];

      // Compute KV range for this split
      const int kv_split_size = div_up(total_kv_len, num_kv_splits);
      const int kv_start = kv_id * kv_split_size;
      const int kv_end = std::min(kv_start + kv_split_size, total_kv_len);

      fill_stub(v_prime_local, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      for (int n = kv_start; n < kv_end; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, kv_end - n);
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        const index_t* __restrict__ indices = req_to_token + req_pool_id * max_context_len + n;
        if (is_mla_shared_kv) {
          pack_vnni_mla<scalar_t, index_t>(
              Btmp0,
              Btmp1,
              k_buffer + head_kv_id * k_strideH,
              indices,
              n_size,
              head_size,
              head_size_v,
              k_strideN,
              BLOCK_N,
              head_size_v);
        } else {
          pack_vnni<scalar_t, index_t>(
              Btmp0,
              k_buffer + head_kv_id * k_strideH,
              indices,
              n_size,
              head_size,
              k_strideN,
              BLOCK_N);
        }

        at::native::cpublas::brgemm(
            m_size, n_size, head_size,
            q_strideM, BLOCK_N, BLOCK_N,
            false, q_ptr, Btmp0, s_i);

        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          float* __restrict__ s_row = s_i + row * BLOCK_N;

          if (has_logit_cap) {
            at::vec::map<float>(
                [scale_vec, logit_cap, rlogit_cap](Vec x) { return apply_logit_cap(x * scale_vec, logit_cap, rlogit_cap); },
                s_row, s_row, n_size);
          } else {
            at::vec::map<float>(
                [scale_vec](Vec x) { return x * scale_vec; }, s_row, s_row, n_size);
          }

          // Apply tree mask
          const bool* mask_row = custom_mask + mask_base + (m + row) * total_kv_len + n;
#if defined(CPU_CAPABILITY_AVX512)
          {
            const Vec neg_inf_vec = Vec(-std::numeric_limits<float>::infinity());
            int col = 0;
            for (; col <= n_size - 16; col += 16) {
              __m128i mask_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask_row + col));
              __mmask16 kmask = _mm_test_epi8_mask(mask_bytes, _mm_set1_epi8(1));
              __m512 vals = _mm512_loadu_ps(s_row + col);
              vals = _mm512_mask_blend_ps(kmask, (__m512)neg_inf_vec, vals);
              _mm512_storeu_ps(s_row + col, vals);
            }
            for (; col < n_size; ++col) {
              if (!mask_row[col]) {
                s_row[col] = -std::numeric_limits<float>::infinity();
              }
            }
          }
#else
          for (int col = 0; col < n_size; ++col) {
            if (!mask_row[col]) {
              s_row[col] = -std::numeric_limits<float>::infinity();
            }
          }
#endif

          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_row, n_size);
          m_i = std::max(m_i, m_prime[row]);

          m_delta[row] = std::exp(m_prime[row] - m_i);

          at::vec::map<float>(
              [m_i](Vec x) { return softmax_exp_u20(x - Vec(m_i)); }, s_delta + row * BLOCK_N, s_row, n_size);

          s_prime[row] *= m_delta[row];
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          float scale_m = m_delta[row];
          at::vec::map<float>(
              [scale_m](Vec x) { return x * Vec(scale_m); },
              v_prime_local + row * head_size_v,
              v_prime_local + row * head_size_v,
              head_size_v);

          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        if (!is_mla_shared_kv) {
          pack_vnni2<scalar_t, index_t>(
              Btmp0,
              v_buffer + head_kv_id * v_strideH,
              indices,
              n_size,
              head_size_v,
              v_strideN,
              head_size_v);
        }

        at::native::cpublas::brgemm(
            m_size, head_size_v, padded_n_size,
            BLOCK_N, head_size_v, head_size_v,
            true, s_delta2, is_mla_shared_kv ? Btmp1 : Btmp0, v_prime_local);
      }  // loop over KV blocks in this split

      // Store partial results into attn_logits
      int work_id = bs * num_heads * MB + head_id * MB + mb;
      float* __restrict__ v_prime_out = attn_logits + work_id * work_stride + kv_id * split_stride;

      if (kv_end > kv_start) {
        for (int row = 0; row < m_size; ++row) {
          if (s_prime[row] > 0.f) {
            float s = 1.f / s_prime[row];
            // Store normalized v_prime
            at::vec::map<float>(
                [s](Vec x) { return x * Vec(s); },
                v_prime_out + row * row_stride,
                v_prime_local + row * head_size_v,
                head_size_v);
            // Store m + log(s) header
            v_prime_out[row * row_stride + head_size_v] = m_prime[row] + std::log(s_prime[row]);
          } else {
            // All KV positions were masked for this row in this split
            fill_stub(v_prime_out + row * row_stride, 0.f, head_size_v);
            v_prime_out[row * row_stride + head_size_v] = -std::numeric_limits<float>::infinity();
          }
        }
        // Mark unused rows as empty
        for (int row = m_size; row < BLOCK_M; ++row) {
          v_prime_out[row * row_stride + head_size_v] = -std::numeric_limits<float>::infinity();
        }
      } else {
        // Empty split
        for (int row = 0; row < BLOCK_M; ++row) {
          fill_stub(v_prime_out + row * row_stride, 0.f, head_size_v);
          v_prime_out[row * row_stride + head_size_v] = -std::numeric_limits<float>::infinity();
        }
      }

      data_index_step(bs, batches, head_id, num_heads, mb, MB, kv_id, num_kv_splits);
    }
    at::native::cpublas::brgemm_release();
  });

  // Merge KV splits and write final output
  extend_accumulate_kv_splits<scalar_t, BLOCK_M>(
      o_extend, attn_logits, work_m_sizes.data(),
      total_work_items, num_kv_splits, head_size_v,
      o_strideM, o_strideH,
      work_seq_start.data(), work_head_ids.data());
}

}  // anonymous namespace

// q_extend, k_extend, v_extend, o_extend: contiguous tensors
// k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
//
// q_extend: [num_tokens, num_heads, head_size]
// k_extend: [num_extend_tokens, num_heads, head_size]
// v_extend: [num_extend_tokens, num_heads, head_size]
// o_extend: [num_tokens, num_heads, head_size]
// k_buffer: [max_total_num_tokens, num_heads, head_size]
// v_buffer: [max_total_num_tokens, num_heads, head_size]
// req_to_token: [max_num_reqs, max_context_len] int32 or int64
// req_pool_indices: [num_seqs] int64
// seq_lens: [num_seqs] int64
// extend_seq_lens: [num_seqs]
// extend_start_loc: [num_seqs]
//
void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::extend_attention_cpu",
      std::vector<c10::IValue>(
          {q_extend,
           k_extend,
           v_extend,
           o_extend,
           k_buffer,
           v_buffer,
           req_to_token,
           req_pool_indices,
           seq_lens,
           extend_seq_lens,
           extend_start_loc}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  // strides for q_extend, k_extend and v_extend
  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  // check sizes
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // D and DV need to be 32x as we transpose by 512-bit
  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);

  // block size for query seq length
  // Tuned for EAGLE speculative decoding (steps=5, topk=4 => draft_tokens=16-20)
  // BLOCK_M=16 reduces padding waste compared to BLOCK_M=32
  constexpr int BLOCK_M = 16;

  int num_threads = at::get_num_threads();
  auto make_buffer = [&](int block_n) {
    const int size_per_thread =
        /* s_i     */ BLOCK_M * block_n * sizeof(float) +
        /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
        /* s_delta */ BLOCK_M * block_n * sizeof(uint16_t) +
        /* Btmp    */ block_n * std::max(head_size, head_size_v) * sizeof(uint16_t);
    return std::make_pair(
        at::empty({num_threads, size_per_thread}, q_extend.options().dtype(at::kChar)),
        size_per_thread);
  };

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_indices", [&] {
      const int selected_block_n = choose_extend_block_n(
          seq_lens.data_ptr<int64_t>(),
          extend_seq_lens.data_ptr<index_t>(),
          num_seqs,
          num_heads_kv,
          head_size,
          is_prefix_skipped);
      if (selected_block_n == 128) {
        auto [buffer, size_per_thread] = make_buffer(128);
        extend_attention_kernel_impl<scalar_t, index_t, BLOCK_M, 128>(
            o_extend.data_ptr<scalar_t>(),
            q_extend.data_ptr<scalar_t>(),
            k_extend.data_ptr<scalar_t>(),
            v_extend.data_ptr<scalar_t>(),
            k_buffer.data_ptr<scalar_t>(),
            v_buffer.data_ptr<scalar_t>(),
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            extend_seq_lens.data_ptr<index_t>(),
            extend_start_loc.data_ptr<index_t>(),
            buffer.data_ptr(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            q_strideM,
            q_strideH,
            ke_strideN,
            ke_strideH,
            ve_strideN,
            ve_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            max_len_extend,
            size_per_thread,
            is_prefix_skipped);
      } else {
        auto [buffer, size_per_thread] = make_buffer(64);
        extend_attention_kernel_impl<scalar_t, index_t, BLOCK_M, 64>(
            o_extend.data_ptr<scalar_t>(),
            q_extend.data_ptr<scalar_t>(),
            k_extend.data_ptr<scalar_t>(),
            v_extend.data_ptr<scalar_t>(),
            k_buffer.data_ptr<scalar_t>(),
            v_buffer.data_ptr<scalar_t>(),
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            extend_seq_lens.data_ptr<index_t>(),
            extend_start_loc.data_ptr<index_t>(),
            buffer.data_ptr(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            q_strideM,
            q_strideH,
            ke_strideN,
            ke_strideH,
            ve_strideN,
            ve_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            max_len_extend,
            size_per_thread,
            is_prefix_skipped);
      }
    });
  });
}

// Tree-mask variant: all KV are read from k_buffer/v_buffer via req_to_token.
// custom_mask: 1-D bool tensor packed per-batch as
//   [draft_len_0 * total_kv_len_0, draft_len_1 * total_kv_len_1, ...]
//
void extend_attention_treemask_cpu(
    at::Tensor& q_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    at::Tensor& custom_mask,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::extend_attention_treemask_cpu",
      std::vector<c10::IValue>(
          {q_extend, o_extend, k_buffer, v_buffer, req_to_token,
           req_pool_indices, seq_lens, extend_seq_lens, extend_start_loc,
           custom_mask}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  TORCH_CHECK(custom_mask.scalar_type() == at::kBool,
      "extend_treemask: expect custom_mask to be bool, got ", custom_mask.scalar_type());
  TORCH_CHECK(custom_mask.is_contiguous(), "extend_treemask: expect custom_mask to be contiguous");

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_buffer.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_buffer.size(2);
  const bool is_mla_shared_kv = is_mla_shared_kv_layout(k_buffer, v_buffer, num_heads_kv);

  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend_treemask: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong,
      "extend_treemask: expect seq_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong,
      "extend_treemask: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend_treemask: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);

  // Same BLOCK_M/BLOCK_N as extend_attention_cpu for consistency
  constexpr int BLOCK_M = 16;

  int num_threads = at::get_num_threads();

  // Compute per-batch mask offsets on CPU
  // mask_offsets[i] = sum_{j<i} (extend_seq_lens[j] * seq_lens[j])
  auto mask_offsets = at::empty({num_seqs}, seq_lens.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_treemask_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_treemask_indices", [&] {
      const int selected_block_n = choose_treemask_block_n(
          seq_lens.data_ptr<int64_t>(),
          extend_seq_lens.data_ptr<index_t>(),
          num_seqs,
          num_heads_kv,
          head_size);
      auto make_buffer = [&](int block_n) {
        const int kv_pack_elems = is_mla_shared_kv ?
            block_n * (head_size + head_size_v) :
            block_n * std::max(head_size, head_size_v);
        const int size_per_thread =
            /* s_i     */ BLOCK_M * block_n * sizeof(float) +
            /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
            /* s_delta */ BLOCK_M * block_n * sizeof(uint16_t) +
            /* Btmp    */ kv_pack_elems * sizeof(uint16_t);
        return std::make_pair(
            at::empty({num_threads, size_per_thread}, q_extend.options().dtype(at::kChar)),
            size_per_thread);
      };

      // Compute mask offsets inside the dispatch block to avoid a separate dispatch
      // In EAGLE verify, seq_lens is the prefix length. The mask is laid out as
      // [extend_len_0 * (prefix_len_0 + extend_len_0), extend_len_1 * (prefix_len_1 + extend_len_1), ...]
      {
        auto seq_lens_ptr = seq_lens.data_ptr<int64_t>();
        auto extend_seq_lens_ptr = extend_seq_lens.data_ptr<index_t>();
        auto mask_offsets_ptr = mask_offsets.data_ptr<int64_t>();
        int64_t offset = 0;
        for (int b = 0; b < num_seqs; ++b) {
          mask_offsets_ptr[b] = offset;
          int64_t total_kv_len_b = seq_lens_ptr[b] + static_cast<int64_t>(extend_seq_lens_ptr[b]);
          offset += static_cast<int64_t>(extend_seq_lens_ptr[b]) * total_kv_len_b;
        }
        TORCH_CHECK(offset == custom_mask.numel(),
            "extend_treemask: custom_mask size mismatch: expected ", offset,
            " but got ", custom_mask.numel());
      }
      // Decide whether to use sequence-parallel path.
      // When batches * num_heads * MB is small relative to num_threads,
      // we split the KV dimension to utilize all cores.
      int MB = div_up(static_cast<int>(max_len_extend), BLOCK_M);
      int base_work = num_seqs * num_heads * MB;
      int num_kv_splits = 1;
      int64_t max_total_kv_len = 0;
      for (int b = 0; b < num_seqs; ++b) {
        max_total_kv_len = std::max<int64_t>(
            max_total_kv_len,
            seq_lens.data_ptr<int64_t>()[b] + static_cast<int64_t>(extend_seq_lens.data_ptr<index_t>()[b]));
      }
      const bool long_gqa_verify = num_heads_kv > 1 && head_size <= 256 && max_total_kv_len >= 4096;

      if (base_work < (long_gqa_verify ? num_threads * 2 : num_threads)) {
        // For long-prefix GQA verify, oversubscribe the KV dimension a bit so bs=1 / MB=1
        // still has enough work to occupy the full socket.
        int target_parallelism = long_gqa_verify ? num_threads * 2 : num_threads;
        num_kv_splits = div_up(target_parallelism, base_work);
        // Clamp to reasonable range: at least 2, at most 32
        num_kv_splits = std::max(num_kv_splits, 2);
        num_kv_splits = std::min(num_kv_splits, long_gqa_verify ? 8 : 32);
      }

    //   std::cout << "extend_attention_treemask_cpu: base_work=" << base_work
    //             << ", MB=" << MB << ", num_seqs=" << num_seqs << ", num_heads=" << num_heads
    //             << ", num_threads=" << num_threads
    //             << ", num_kv_splits=" << num_kv_splits << std::endl;

      if (num_kv_splits > 1) {
        // Sequence-parallel path: split KV dimension across threads
        // Allocate attn_logits for intermediate partial results
        // Layout: [base_work, num_kv_splits, BLOCK_M * (head_size_v + 1)]
        int row_stride = head_size_v + 1;
        int split_stride = BLOCK_M * row_stride;
        int64_t logits_size = static_cast<int64_t>(base_work) * num_kv_splits * split_stride;
        auto attn_logits = at::empty({logits_size}, q_extend.options().dtype(at::kFloat));
        if (selected_block_n == 128) {
          auto [buffer, size_per_thread] = make_buffer(128);
          extend_attention_treemask_sp_kernel_impl<scalar_t, index_t, BLOCK_M, 128>(
              o_extend.data_ptr<scalar_t>(),
              q_extend.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<index_t>(),
              extend_start_loc.data_ptr<index_t>(),
              custom_mask.data_ptr<bool>(),
              mask_offsets.data_ptr<int64_t>(),
              attn_logits.data_ptr<float>(),
              buffer.data_ptr(),
              num_seqs,
              num_heads,
              num_heads_kv,
              head_size,
              head_size_v,
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
              max_len_extend,
              num_kv_splits,
              is_mla_shared_kv,
              size_per_thread);
        } else {
          auto [buffer, size_per_thread] = make_buffer(64);
          extend_attention_treemask_sp_kernel_impl<scalar_t, index_t, BLOCK_M, 64>(
              o_extend.data_ptr<scalar_t>(),
              q_extend.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<index_t>(),
              extend_start_loc.data_ptr<index_t>(),
              custom_mask.data_ptr<bool>(),
              mask_offsets.data_ptr<int64_t>(),
              attn_logits.data_ptr<float>(),
              buffer.data_ptr(),
              num_seqs,
              num_heads,
              num_heads_kv,
              head_size,
              head_size_v,
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
              max_len_extend,
              num_kv_splits,
              is_mla_shared_kv,
              size_per_thread);
        }
      } else {
        // Original non-split path: sufficient parallelism already
        if (selected_block_n == 128) {
          auto [buffer, size_per_thread] = make_buffer(128);
          extend_attention_treemask_kernel_impl<scalar_t, index_t, BLOCK_M, 128>(
              o_extend.data_ptr<scalar_t>(),
              q_extend.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<index_t>(),
              extend_start_loc.data_ptr<index_t>(),
              custom_mask.data_ptr<bool>(),
              mask_offsets.data_ptr<int64_t>(),
              buffer.data_ptr(),
              num_seqs,
              num_heads,
              num_heads_kv,
              head_size,
              head_size_v,
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
              max_len_extend,
              is_mla_shared_kv,
              size_per_thread);
        } else {
          auto [buffer, size_per_thread] = make_buffer(64);
          extend_attention_treemask_kernel_impl<scalar_t, index_t, BLOCK_M, 64>(
              o_extend.data_ptr<scalar_t>(),
              q_extend.data_ptr<scalar_t>(),
              k_buffer.data_ptr<scalar_t>(),
              v_buffer.data_ptr<scalar_t>(),
              req_to_token.data_ptr<index_t>(),
              req_pool_indices.data_ptr<int64_t>(),
              seq_lens.data_ptr<int64_t>(),
              extend_seq_lens.data_ptr<index_t>(),
              extend_start_loc.data_ptr<index_t>(),
              custom_mask.data_ptr<bool>(),
              mask_offsets.data_ptr<int64_t>(),
              buffer.data_ptr(),
              num_seqs,
              num_heads,
              num_heads_kv,
              head_size,
              head_size_v,
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
              max_len_extend,
              is_mla_shared_kv,
              size_per_thread);
        }
      }
    });
  });
}
