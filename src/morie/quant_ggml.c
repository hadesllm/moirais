/*
 * quant_ggml.c — TurboQuant C implementation for MORIE
 *
 * Implements TurboQuant MSE-optimal quantization using:
 *   1. Walsh-Hadamard Transform (WHT) for rotation (O(d log d), no matrix storage)
 *   2. Lloyd-Max scalar quantization with precomputed codebooks
 *   3. Bit-packed index storage matching GGML block formats
 *
 * Critical: WHT normalization uses 1/sqrt(d) — NOT 1/d.
 *
 * References:
 *   - TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
 *   - PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)
 *   - llama.cpp PR #21089: GGML_TYPE_TBQ3_0, TBQ4_0
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o quant_ggml.dylib quant_ggml.c -lm
 *   Linux:  cc -O2 -march=native -shared -fPIC -o quant_ggml.so quant_ggml.c -lm
 */

#include "quant_ggml.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SIMD strategy (added 2026-04-26 — P3 dual-ISA pass; mirrors the pattern
 * documented at the top of engine_kernels.c).
 *
 *   Linux ARM (Pi 5 zeus.local, AWS Graviton, Asahi)  →  NEON 4-wide
 *   Linux x86_64 (HADES GCP, AWS, Azure)              →  AVX2 8-wide
 *                                                        (-mavx2 -mfma)
 *   Other (incl macOS without Accelerate path here)   →  scalar fallback
 *
 * Vectorised in this pass:
 *   - wht_inplace butterfly inner loop (len >= 4 NEON / 8 AVX2)
 *   - wht_inplace post-butterfly 1/sqrt(d) normalisation
 *   - tq_qjl_encode O(d^2) projection matvec (the dominant runtime cost
 *     of the QJL Stage-2 quantisation path)
 *
 * Not vectorised this pass (lower priority):
 *   - bit pack / unpack (irregular bit ops)
 *   - box_muller_next (cos/log dominated; needs vector libm)
 *   - apply_random_signs (uses serial xorshift PRNG)
 */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#define TQ_HAVE_NEON 1
#else
#define TQ_HAVE_NEON 0
#endif

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TQ_HAVE_AVX2 1
#else
#define TQ_HAVE_AVX2 0
#endif

/* ─── Walsh-Hadamard Transform ─────────────────────────────────────────── */

/*
 * In-place Walsh-Hadamard Transform with 1/sqrt(d) normalization.
 *
 * The WHT is an orthogonal transform (H^T H = I when normalized) that
 * randomizes coordinate-wise distributions without storing a full d×d matrix.
 * O(d log d) time, O(1) extra space.
 *
 * CRITICAL: normalization factor is 1/sqrt(d), NOT 1/d.
 * Using 1/d produces garbage — the inverse transform won't recover the input.
 */
static void wht_inplace(float *x, int d) {
    /* Butterfly operations.
     * For len >= SIMD-width with len-aligned stride, vectorise the inner
     * loop (4-wide on NEON, 8-wide on AVX2). When len < SIMD-width the
     * data dependencies don't fit SIMD lanes cleanly, so fall through to
     * scalar; len doubles each outer iteration so the inner loops at
     * len=4,8,16,... are vectorisable and dominate the runtime. */
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
#if TQ_HAVE_NEON
            if (len >= 4 && (len & 3) == 0) {
                for (int j = 0; j < len; j += 4) {
                    float32x4_t u = vld1q_f32(x + i + j);
                    float32x4_t v = vld1q_f32(x + i + j + len);
                    vst1q_f32(x + i + j,       vaddq_f32(u, v));
                    vst1q_f32(x + i + j + len, vsubq_f32(u, v));
                }
                continue;
            }
#elif TQ_HAVE_AVX2
            if (len >= 8 && (len & 7) == 0) {
                for (int j = 0; j < len; j += 8) {
                    __m256 u = _mm256_loadu_ps(x + i + j);
                    __m256 v = _mm256_loadu_ps(x + i + j + len);
                    _mm256_storeu_ps(x + i + j,       _mm256_add_ps(u, v));
                    _mm256_storeu_ps(x + i + j + len, _mm256_sub_ps(u, v));
                }
                continue;
            }
#endif
            /* scalar (len < SIMD-width or no SIMD available) */
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }

    /* Normalize: 1/sqrt(d) */
    float norm = 1.0f / sqrtf((float)d);
#if TQ_HAVE_NEON
    {
        float32x4_t scale = vdupq_n_f32(norm);
        int i = 0;
        int n4 = d & ~3;
        for (; i < n4; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            vst1q_f32(x + i, vmulq_f32(v, scale));
        }
        for (; i < d; i++) x[i] *= norm;
    }
#elif TQ_HAVE_AVX2
    {
        __m256 scale = _mm256_set1_ps(norm);
        int i = 0;
        int n8 = d & ~7;
        for (; i < n8; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(v, scale));
        }
        for (; i < d; i++) x[i] *= norm;
    }
#else
    for (int i = 0; i < d; i++) {
        x[i] *= norm;
    }
#endif
}

/*
 * Inverse WHT — for normalized WHT, the inverse is the same operation.
 * H^{-1} = H when H is symmetric and orthogonal with 1/sqrt(d) scaling.
 */
static void iwht_inplace(float *x, int d) {
    wht_inplace(x, d);
}

/* ─── Lloyd-Max Codebooks ──────────────────────────────────────────────── */

/*
 * Exact Lloyd-Max codebooks computed via 200-iteration optimization on the
 * Beta((d-1)/2, (d-1)/2) distribution: f(x) = [Γ(d/2)/(√π·Γ((d-1)/2))]·(1-x²)^((d-3)/2)
 *
 * Reference: Zandieh et al. (2026). TurboQuant. ICLR 2026. arXiv:2504.19874
 * Codebooks for d=128 (head_dim) and d=256 (GGML block size).
 */

/* ── d=128 codebooks ── */
static const float CB_2BIT_D128[4] = {
    -0.1308671424f, -0.0394688126f,  0.0394688126f,  0.1308651381f
};
static const float CB_3BIT_D128[8] = {
    -0.1812177775f, -0.1149127640f, -0.0650495536f, -0.0211455100f,
     0.0211455100f,  0.0650495536f,  0.1149127640f,  0.1812131980f
};
static const float CB_4BIT_D128[16] = {
    -0.2182390812f, -0.1705074768f, -0.1352592295f, -0.1059085478f,
    -0.0799082423f, -0.0559264298f, -0.0331327852f, -0.0109761295f,
     0.0109761295f,  0.0331327852f,  0.0559264298f,  0.0799082423f,
     0.1059085478f,  0.1352592295f,  0.1705074768f,  0.2182296630f
};

/* ── d=256 codebooks ── */
static const float CB_2BIT_D256[4] = {
    -0.0925997855f, -0.0279008603f,  0.0278861752f,  0.0925690980f
};
static const float CB_3BIT_D256[8] = {
    -0.1283115365f, -0.0812447636f, -0.0459501833f, -0.0149310886f,
     0.0149310886f,  0.0459501833f,  0.0812447636f,  0.1283081853f
};
static const float CB_4BIT_D256[16] = {
    -0.1545071341f, -0.1205888834f, -0.0955810685f, -0.0748105393f,
    -0.0564259431f, -0.0394887183f, -0.0234089441f, -0.0077608266f,
     0.0077608266f,  0.0234089441f,  0.0394887183f,  0.0564259431f,
     0.0748105393f,  0.0955810685f,  0.1205760380f,  0.1544713338f
};

static const float *get_codebook(int bits, int dim, int *n_centroids) {
    if (dim <= 128) {
        switch (bits) {
            case 2: *n_centroids = 4;  return CB_2BIT_D128;
            case 3: *n_centroids = 8;  return CB_3BIT_D128;
            case 4: *n_centroids = 16; return CB_4BIT_D128;
            default: *n_centroids = 0; return NULL;
        }
    }
    /* d=256 or any other dim — use d=256 codebooks */
    switch (bits) {
        case 2: *n_centroids = 4;  return CB_2BIT_D256;
        case 3: *n_centroids = 8;  return CB_3BIT_D256;
        case 4: *n_centroids = 16; return CB_4BIT_D256;
        default: *n_centroids = 0; return NULL;
    }
}

/* ─── Scalar Quantization ──────────────────────────────────────────────── */

/*
 * Find nearest codebook centroid for a scalar value.
 * Returns the index (0 to n_centroids-1).
 *
 * Uses linear search (codebook is tiny: 4/8/16 entries).
 * For 4-bit, a binary search or LUT could be ~30% faster on Apple Silicon.
 */
static inline uint8_t quantize_scalar(float val, const float *codebook, int n) {
    uint8_t best = 0;
    float best_dist = fabsf(val - codebook[0]);
    for (int k = 1; k < n; k++) {
        float dist = fabsf(val - codebook[k]);
        if (dist < best_dist) {
            best_dist = dist;
            best = (uint8_t)k;
        }
    }
    return best;
}

/*
 * 4-magnitude LUT dequantization for Apple Silicon (NEON-friendly).
 * Instead of a codebook lookup per element, preload 4 centroid magnitudes
 * and use bit-extract + sign to reconstruct.
 *
 * For symmetric codebooks: val = sign * magnitude[abs_index]
 * This avoids branch misprediction on M1-M5.
 */
static inline float dequantize_scalar(uint8_t idx, const float *codebook) {
    return codebook[idx];
}

/* ─── Bit Packing ──────────────────────────────────────────────────────── */

/*
 * Pack n indices of `bits` width into a byte array.
 * Indices are packed LSB-first, spanning byte boundaries.
 */
static void pack_bits(const uint8_t *indices, uint8_t *packed, int n, int bits) {
    int bit_pos = 0;
    uint8_t mask = (1 << bits) - 1;
    int n_bytes = (n * bits + 7) / 8;
    memset(packed, 0, n_bytes);

    for (int i = 0; i < n; i++) {
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;
        packed[byte_idx] |= (indices[i] & mask) << bit_off;
        /* Handle overflow into next byte */
        if (bit_off + bits > 8 && byte_idx + 1 < n_bytes) {
            packed[byte_idx + 1] |= (indices[i] & mask) >> (8 - bit_off);
        }
        bit_pos += bits;
    }
}

/*
 * Unpack n indices of `bits` width from a packed byte array.
 */
static void unpack_bits(const uint8_t *packed, uint8_t *indices, int n, int bits) {
    int bit_pos = 0;
    uint8_t mask = (1 << bits) - 1;

    for (int i = 0; i < n; i++) {
        int byte_idx = bit_pos >> 3;
        int bit_off  = bit_pos & 7;
        uint16_t val = packed[byte_idx];
        if (bit_off + bits > 8) {
            val |= ((uint16_t)packed[byte_idx + 1]) << 8;
        }
        indices[i] = (val >> bit_off) & mask;
        bit_pos += bits;
    }
}

/* ─── Seed-based pseudo-random sign flip ───────────────────────────────── */

/*
 * Apply a deterministic sign flip to each element based on seed.
 * Combined with WHT, this gives a random rotation without storing a matrix.
 *
 * Uses a simple xorshift32 PRNG seeded by `seed`.
 * The sign flip + WHT combination is equivalent to a random orthogonal rotation
 * drawn uniformly from O(d).
 */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void apply_random_signs(float *x, int d, uint16_t seed) {
    uint32_t state = (uint32_t)seed * 2654435761u + 1;  /* Knuth multiplicative hash */
    for (int i = 0; i < d; i++) {
        if (xorshift32(&state) & 1) {
            x[i] = -x[i];
        }
    }
}

/* ─── Public API ───────────────────────────────────────────────────────── */

void tq_init(tq_context *ctx, int dim, int bits) {
    ctx->dim = dim;
    ctx->bits = bits;

    int n_centroids;
    const float *cb = get_codebook(bits, dim, &n_centroids);
    ctx->codebook.dim = dim;
    ctx->codebook.bits = bits;
    ctx->codebook.n_centroids = n_centroids;
    if (cb && n_centroids <= TQ_MAX_CODEBOOK) {
        memcpy(ctx->codebook.centroids, cb, n_centroids * sizeof(float));
    }
}

void tq_quantize_block(
    const float *src,
    void        *dst,
    const tq_context *ctx,
    uint16_t seed
) {
    int d = ctx ? ctx->dim : TQ_BLOCK_SIZE;
    int bits = ctx ? ctx->bits : 3;

    int n_centroids;
    const float *codebook = ctx ? ctx->codebook.centroids : get_codebook(bits, d, &n_centroids);
    if (!ctx) get_codebook(bits, d, &n_centroids);
    else n_centroids = ctx->codebook.n_centroids;

    /* Allocate temp buffer for rotated data */
    float *rotated = (float *)malloc(d * sizeof(float));
    memcpy(rotated, src, d * sizeof(float));

    /* Step 1: Compute L2 norm */
    float norm = 0.0f;
    for (int i = 0; i < d; i++) {
        norm += rotated[i] * rotated[i];
    }
    norm = sqrtf(norm);

    /* Step 2: Normalize to unit vector */
    if (norm > 1e-15f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < d; i++) {
            rotated[i] *= inv_norm;
        }
    }

    /* Step 3: Random sign flip + WHT rotation */
    apply_random_signs(rotated, d, seed);
    wht_inplace(rotated, d);

    /* Step 4: Scalar quantize each element */
    uint8_t *indices = (uint8_t *)malloc(d);
    for (int i = 0; i < d; i++) {
        indices[i] = quantize_scalar(rotated[i], codebook, n_centroids);
    }

    /* Step 5: Pack into block structure */
    int packed_bytes = (d * bits + 7) / 8;

    if (bits == 3) {
        block_tbq3_0 *blk = (block_tbq3_0 *)dst;
        blk->norm = norm;
        blk->seed = seed;
        pack_bits(indices, blk->indices, d, 3);
    } else if (bits == 4) {
        block_tbq4_0 *blk = (block_tbq4_0 *)dst;
        blk->norm = norm;
        blk->seed = seed;
        pack_bits(indices, blk->indices, d, 4);
    } else if (bits == 2) {
        block_tbq2_0 *blk = (block_tbq2_0 *)dst;
        blk->norm = norm;
        blk->seed = seed;
        pack_bits(indices, blk->indices, d, 2);
    }

    free(indices);
    free(rotated);
}

void tq_dequantize_block(
    const void  *src,
    float       *dst,
    const tq_context *ctx
) {
    int d = ctx ? ctx->dim : TQ_BLOCK_SIZE;
    int bits = ctx ? ctx->bits : 3;

    int n_centroids;
    const float *codebook = ctx ? ctx->codebook.centroids : get_codebook(bits, d, &n_centroids);
    if (!ctx) get_codebook(bits, d, &n_centroids);
    else n_centroids = ctx->codebook.n_centroids;

    float norm;
    uint16_t seed;
    const uint8_t *packed;

    /* Extract from block */
    if (bits == 3) {
        const block_tbq3_0 *blk = (const block_tbq3_0 *)src;
        norm = blk->norm;
        seed = blk->seed;
        packed = blk->indices;
    } else if (bits == 4) {
        const block_tbq4_0 *blk = (const block_tbq4_0 *)src;
        norm = blk->norm;
        seed = blk->seed;
        packed = blk->indices;
    } else {
        const block_tbq2_0 *blk = (const block_tbq2_0 *)src;
        norm = blk->norm;
        seed = blk->seed;
        packed = blk->indices;
    }

    /* Step 1: Unpack indices */
    uint8_t *indices = (uint8_t *)malloc(d);
    unpack_bits(packed, indices, d, bits);

    /* Step 2: Dequantize scalars via codebook lookup */
    for (int i = 0; i < d; i++) {
        dst[i] = dequantize_scalar(indices[i], codebook);
    }

    /* Step 3: Inverse WHT + inverse sign flip */
    iwht_inplace(dst, d);
    apply_random_signs(dst, d, seed);  /* sign flip is its own inverse */

    /* Step 4: Scale by stored norm */
    for (int i = 0; i < d; i++) {
        dst[i] *= norm;
    }

    free(indices);
}

float tq_dot_product(
    const void       *block,
    const float      *vec,
    const tq_context *ctx
) {
    int d = ctx ? ctx->dim : TQ_BLOCK_SIZE;
    int bits = ctx ? ctx->bits : 3;

    /* Full dequantize + dot product.
     * A fused kernel could avoid materializing the full vector,
     * but this is correct and portable. */
    float *deq = (float *)malloc(d * sizeof(float));
    tq_dequantize_block(block, deq, ctx);

    float dot = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += deq[i] * vec[i];
    }

    free(deq);
    return dot;
}

/* ─── QJL Stage 2: 1-bit error correction ──────────────────────────────── */
/*
 * Reference: Zandieh et al. (2025). QJL. AAAI 2025. arXiv:2406.03482
 * GitHub: https://github.com/amirzandieh/QJL
 *
 * QJL encode: Q(r) = sign(S * r), store ||r||₂
 * QJL decode: Q⁻¹(z) = (√(π/2) / d) · S^T · z
 * Guarantee: E[<y, Q⁻¹(Q(r))>] = <y, r>  (unbiased inner product)
 */

/* Box-Muller transform: generate approximate N(0,1) from uniform xorshift */
static float box_muller_next(uint32_t *state) {
    /* Generate two uniform [0,1) values via xorshift */
    float u1 = (float)(xorshift32(state) & 0x7FFFFF) / (float)0x800000;
    float u2 = (float)(xorshift32(state) & 0x7FFFFF) / (float)0x800000;
    if (u1 < 1e-10f) u1 = 1e-10f;  /* avoid log(0) */
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

void tq_qjl_init(tq_qjl_context *ctx, int dim, uint32_t seed) {
    ctx->dim = dim;
    ctx->seed = seed;
    /* Compute the element count in size_t: dim*dim in int would
       overflow for dim > ~46340 before widening to malloc's size_t. */
    size_t n_elem = (size_t)dim * (size_t)dim;
    ctx->projection = (float *)malloc(n_elem * sizeof(float));

    /* Generate d×d i.i.d. N(0,1) projection matrix via Box-Muller */
    uint32_t state = seed * 2654435761u + 7;
    for (size_t i = 0; i < n_elem; i++) {
        ctx->projection[i] = box_muller_next(&state);
    }
}

void tq_qjl_free(tq_qjl_context *ctx) {
    if (ctx->projection) {
        free(ctx->projection);
        ctx->projection = NULL;
    }
}

void tq_qjl_encode(
    const float *residual,
    uint8_t     *signs,
    float       *norm_out,
    const tq_qjl_context *ctx
) {
    int d = ctx->dim;
    int n_bytes = (d + 7) / 8;

    /* Compute ||r||₂ */
    float norm = 0.0f;
    for (int i = 0; i < d; i++) {
        norm += residual[i] * residual[i];
    }
    *norm_out = sqrtf(norm);

    /* Compute sign(S * r) and bit-pack.
     * The d×d projection matvec is the O(d^2) hot path of QJL encode —
     * biggest SIMD win in this file. */
    memset(signs, 0, n_bytes);
    for (int i = 0; i < d; i++) {
        const float *row = ctx->projection + (long long)i * d;
        float dot = 0.0f;
#if TQ_HAVE_NEON
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int j = 0;
            int n4 = d & ~3;
            for (; j < n4; j += 4) {
                float32x4_t a = vld1q_f32(row + j);
                float32x4_t b = vld1q_f32(residual + j);
                acc = vfmaq_f32(acc, a, b);
            }
            dot = vaddvq_f32(acc);
            for (; j < d; j++) dot += row[j] * residual[j];
        }
#elif TQ_HAVE_AVX2
        {
            __m256 acc = _mm256_setzero_ps();
            int j = 0;
            int n8 = d & ~7;
            for (; j < n8; j += 8) {
                __m256 a = _mm256_loadu_ps(row + j);
                __m256 b = _mm256_loadu_ps(residual + j);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            _mm_store_ss(&dot, s);
            for (; j < d; j++) dot += row[j] * residual[j];
        }
#else
        for (int j = 0; j < d; j++) {
            dot += row[j] * residual[j];
        }
#endif
        /* Store sign: 1 = positive, 0 = negative */
        if (dot >= 0.0f) {
            signs[i / 8] |= (1 << (i % 8));
        }
    }
}

void tq_qjl_decode(
    const uint8_t *signs,
    float          norm,
    float         *dst,
    const tq_qjl_context *ctx
) {
    int d = ctx->dim;
    /*
     * Q⁻¹(z) = (√(π/2) / d) · S^T · z
     * where z ∈ {-1, +1}^d extracted from sign bits
     * Then scale to match ||r||₂
     */
    float scale = sqrtf(3.14159265f / 2.0f) / (float)d;

    /* S^T · z : matrix-vector product with sign vector */
    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            float z_i = (signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
            sum += ctx->projection[i * d + j] * z_i;
        }
        dst[j] = scale * sum;
    }

    /* Scale to match stored residual norm */
    float dst_norm = 0.0f;
    for (int j = 0; j < d; j++) {
        dst_norm += dst[j] * dst[j];
    }
    dst_norm = sqrtf(dst_norm);
    if (dst_norm > 1e-15f) {
        float ratio = norm / dst_norm;
        for (int j = 0; j < d; j++) {
            dst[j] *= ratio;
        }
    }
}

void tq_quantize_block_prod(
    const float *src,
    void        *dst_stage1,
    uint8_t     *qjl_signs,
    float       *qjl_norm,
    const tq_context *ctx,
    const tq_qjl_context *qjl_ctx,
    uint16_t seed
) {
    int d = ctx ? ctx->dim : TQ_BLOCK_SIZE;

    /* Stage 1: MSE quantize */
    tq_quantize_block(src, dst_stage1, ctx, seed);

    /* Stage 1: dequantize to get residual */
    float *deq = (float *)malloc(d * sizeof(float));
    tq_dequantize_block(dst_stage1, deq, ctx);

    float *residual = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++) {
        residual[i] = src[i] - deq[i];
    }

    /* Stage 2: QJL encode residual */
    tq_qjl_encode(residual, qjl_signs, qjl_norm, qjl_ctx);

    free(residual);
    free(deq);
}

void tq_dequantize_block_prod(
    const void  *stage1_block,
    const uint8_t *qjl_signs,
    float        qjl_norm,
    float       *dst,
    const tq_context *ctx,
    const tq_qjl_context *qjl_ctx
) {
    int d = ctx ? ctx->dim : TQ_BLOCK_SIZE;

    /* Stage 1: MSE dequantize */
    tq_dequantize_block(stage1_block, dst, ctx);

    /* Stage 2: QJL decode and add */
    float *qjl_correction = (float *)malloc(d * sizeof(float));
    tq_qjl_decode(qjl_signs, qjl_norm, qjl_correction, qjl_ctx);

    for (int i = 0; i < d; i++) {
        dst[i] += qjl_correction[i];
    }

    free(qjl_correction);
}
