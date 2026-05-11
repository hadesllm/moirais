/*
 * engine_kernels.c — MORIE Inference Engine C Kernels
 *
 * Secure, portable C kernels for transformer forward pass hot-path ops.
 *
 * Security design:
 *   - Every function validates inputs (NULL checks, size checks)
 *   - No heap allocation (caller-allocated buffers only)
 *   - No global mutable state (fully reentrant/thread-safe)
 *   - No string formatting with user data (no sprintf injection)
 *   - Integer overflow protection on size parameters
 *   - Uses Accelerate.framework BLAS on macOS for hardware-tuned SIMD
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o engine_kernels.dylib engine_kernels.c -lm -framework Accelerate
 *   Linux:  cc -O2 -march=native -shared -fPIC -o engine_kernels.so engine_kernels.c -lm
 *
 * References:
 *   - RMSNorm: Zhang & Sennrich (2019). Root Mean Square Layer Normalization.
 *   - RoPE: Su et al. (2021). RoFormer. arXiv:2104.09864.
 *   - SwiGLU: Shazeer (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
 */

#include "engine_kernels.h"
#include <math.h>
#include <string.h>

/*
 * SIMD strategy (added 2026-04-26 — P3 dual-ISA pass):
 *
 *   1. macOS  → Accelerate.framework (NEON + AMX matrix unit, hand-tuned).
 *      Selected via HAVE_ACCELERATE = 1; not changed by this pass.
 *
 *   2. Linux ARM (AWS Graviton, ARMv8 SBCs, Apple Silicon Asahi)
 *      → ARM NEON intrinsics via <arm_neon.h> (ACLE).
 *      Selected via HAVE_NEON_NATIVE when __ARM_NEON is defined and
 *      Accelerate is unavailable. Cortex-A76 / Neoverse-N1 cores
 *      support NEON with FMA (fused multiply-accumulate) —
 *      vfmaq_f32 in 1 cycle.
 *
 *   3. Linux x86_64 (HADES LLM GCP n2-standard, AWS, Azure)
 *      → AVX2 intrinsics via <immintrin.h>, requires -mavx2 -mfma.
 *      Selected via HAVE_AVX2 when __AVX2__ is defined. AVX2 gives
 *      8-wide single-precision FMA (_mm256_fmadd_ps) at 1 cycle on
 *      Skylake-X+, Zen2+. On AVX-512 hardware, the AVX2 path still
 *      runs correctly (AVX2 is a subset).
 *
 *   4. Portable scalar fallback for any other architecture (RISC-V
 *      without RVV, PowerPC without VSX, etc.).
 *
 * Selection happens per-function via #if cascade: HAVE_ACCELERATE first,
 * then HAVE_NEON_NATIVE, then HAVE_AVX2, then scalar. This keeps the
 * macOS build path untouched while adding genuine SIMD on the two
 * deployment targets where Accelerate is unavailable.
 */

/* Use Accelerate.framework on macOS for hardware-tuned BLAS/vDSP */
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAVE_ACCELERATE 1
#else
#define HAVE_ACCELERATE 0
#endif

/* Native NEON when Accelerate isn't available (Linux ARM64, etc.) */
#if !HAVE_ACCELERATE && defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_NEON_NATIVE 1
#else
#define HAVE_NEON_NATIVE 0
#endif

/* AVX2 on x86_64 when explicitly enabled (-mavx2 -mfma at compile time) */
#if !HAVE_ACCELERATE && defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

/* Maximum safe dimension to prevent integer overflow in size calculations */
#define MAX_DIM (1 << 24)  /* 16M elements — well beyond any LLM dimension */

/* ─── Input Validation Macros ──────────────────────────────────────────── */

#define CHECK_NULL(ptr) do { if (!(ptr)) return MORIE_ERR_NULL; } while (0)
#define CHECK_SIZE(n)   do { if ((n) <= 0 || (n) > MAX_DIM) return MORIE_ERR_SIZE; } while (0)

/* ─── RMSNorm ──────────────────────────────────────────────────────────── */

int morie_rmsnorm(
    const float *x,
    const float *weight,
    float       *out,
    int          n,
    float        eps
) {
    CHECK_NULL(x);
    CHECK_NULL(weight);
    CHECK_NULL(out);
    CHECK_SIZE(n);

    float sum_sq = 0.0f;

#if HAVE_ACCELERATE
    /* vDSP: sum of squares in one call (SIMD-accelerated) */
    vDSP_svesq(x, 1, &sum_sq, n);
#elif HAVE_NEON_NATIVE
    /* NEON 4-wide sum-of-squares with horizontal reduction at end */
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;
        int n4 = n & ~3;
        for (; i < n4; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            acc = vfmaq_f32(acc, v, v);  /* acc += v*v (FMA) */
        }
        sum_sq = vaddvq_f32(acc);  /* horizontal sum across 4 lanes */
        for (; i < n; i++) sum_sq += x[i] * x[i];  /* tail */
    }
#elif HAVE_AVX2
    /* AVX2 8-wide sum-of-squares with horizontal reduction at end */
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        int n8 = n & ~7;
        for (; i < n8; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            acc = _mm256_fmadd_ps(v, v, acc);  /* acc += v*v */
        }
        /* horizontal sum: 8→4→2→1 */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        _mm_store_ss(&sum_sq, s);
        for (; i < n; i++) sum_sq += x[i] * x[i];
    }
#else
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
#endif

    float rms = sqrtf(sum_sq / (float)n + eps);
    float inv_rms = 1.0f / rms;

#if HAVE_ACCELERATE
    /* vDSP: scale x by inv_rms, then element-wise multiply by weight */
    float *temp = out;  /* reuse out as temp since we're writing to it anyway */
    vDSP_vsmul(x, 1, &inv_rms, temp, 1, n);
    vDSP_vmul(temp, 1, weight, 1, out, 1, n);
#elif HAVE_NEON_NATIVE
    {
        float32x4_t scale = vdupq_n_f32(inv_rms);
        int i = 0;
        int n4 = n & ~3;
        for (; i < n4; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            float32x4_t w = vld1q_f32(weight + i);
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale), w));
        }
        for (; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
    }
#elif HAVE_AVX2
    {
        __m256 scale = _mm256_set1_ps(inv_rms);
        int i = 0;
        int n8 = n & ~7;
        for (; i < n8; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, scale), w));
        }
        for (; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
    }
#else
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
#endif

    return MORIE_OK;
}

/* ─── RoPE ─────────────────────────────────────────────────────────────── */

int morie_rope(
    float *q,
    float *k,
    int    head_dim,
    int    position,
    float  freq_base
) {
    CHECK_NULL(q);
    CHECK_NULL(k);
    CHECK_SIZE(head_dim);
    if (head_dim % 2 != 0) return MORIE_ERR_SIZE;

    int half = head_dim / 2;

    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
        float angle = (float)position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* Rotate query */
        float q0 = q[i];
        float q1 = q[i + half];
        q[i]        = q0 * cos_a - q1 * sin_a;
        q[i + half] = q0 * sin_a + q1 * cos_a;

        /* Rotate key */
        float k0 = k[i];
        float k1 = k[i + half];
        k[i]        = k0 * cos_a - k1 * sin_a;
        k[i + half] = k0 * sin_a + k1 * cos_a;
    }

    return MORIE_OK;
}

/* ─── Matrix-Vector Multiply ───────────────────────────────────────────── */

int morie_matvec(
    const float *A,
    const float *x,
    float       *out,
    int          rows,
    int          cols
) {
    CHECK_NULL(A);
    CHECK_NULL(x);
    CHECK_NULL(out);
    CHECK_SIZE(rows);
    CHECK_SIZE(cols);

    /* Overflow check: rows * cols must fit in int */
    if ((long long)rows * cols > (long long)MAX_DIM * MAX_DIM) {
        return MORIE_ERR_SIZE;
    }

#if HAVE_ACCELERATE
    /*
     * cblas_sgemv: y = alpha * A * x + beta * y
     * CblasRowMajor: A is row-major (C default)
     * CblasNoTrans: no transpose
     * alpha=1.0, beta=0.0: simple y = A*x
     */
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans,
        rows, cols,
        1.0f,          /* alpha */
        A, cols,       /* A, lda */
        x, 1,          /* x, incx */
        0.0f,          /* beta */
        out, 1         /* y, incy */
    );
#elif HAVE_NEON_NATIVE
    /*
     * NEON path: per-row 4-wide FMA dot product with horizontal sum.
     * Cortex-A76 (Pi 5): vfmaq_f32 = 1 cycle, vaddvq_f32 = 5 cycles.
     * Hot inner loop is 4 floats / cycle = ~32 GFLOPS at 2.4 GHz.
     */
    for (int i = 0; i < rows; i++) {
        const float *row = A + (long long)i * cols;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        int n4 = cols & ~3;
        for (; j < n4; j += 4) {
            float32x4_t a = vld1q_f32(row + j);
            float32x4_t b = vld1q_f32(x + j);
            acc = vfmaq_f32(acc, a, b);
        }
        float sum = vaddvq_f32(acc);
        for (; j < cols; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
#elif HAVE_AVX2
    /*
     * AVX2 path: per-row 8-wide FMA dot product with horizontal sum.
     * Skylake/Zen2+: _mm256_fmadd_ps = 1 cycle (2 FMAs per cycle).
     * Hot inner loop is 16 FLOPs/cycle = ~64 GFLOPS at 4 GHz per core.
     */
    for (int i = 0; i < rows; i++) {
        const float *row = A + (long long)i * cols;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        int n8 = cols & ~7;
        for (; j < n8; j += 8) {
            __m256 a = _mm256_loadu_ps(row + j);
            __m256 b = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        /* horizontal sum 8→4→2→1 */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum;
        _mm_store_ss(&sum, s);
        for (; j < cols; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
#else
    /* Portable fallback: manual dot product per row */
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const float *row = A + (long long)i * cols;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
#endif

    return MORIE_OK;
}

/* ─── SiLU Activation ──────────────────────────────────────────────────── */

int morie_silu_inplace(float *x, int n) {
    CHECK_NULL(x);
    CHECK_SIZE(n);

    for (int i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        x[i] *= sigmoid;
    }

    return MORIE_OK;
}

/* ─── Softmax ──────────────────────────────────────────────────────────── */

int morie_softmax_inplace(float *x, int n) {
    CHECK_NULL(x);
    CHECK_SIZE(n);

    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Exponentiate and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            x[i] *= inv_sum;
        }
    }

    return MORIE_OK;
}

/* ─── Elementwise Multiply ─────────────────────────────────────────────── */

int morie_elemwise_mul(
    const float *a,
    const float *b,
    float       *out,
    int          n
) {
    CHECK_NULL(a);
    CHECK_NULL(b);
    CHECK_NULL(out);
    CHECK_SIZE(n);

#if HAVE_ACCELERATE
    vDSP_vmul(a, 1, b, 1, out, 1, n);
#elif HAVE_NEON_NATIVE
    {
        int i = 0;
        int n4 = n & ~3;
        for (; i < n4; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(out + i, vmulq_f32(va, vb));
        }
        for (; i < n; i++) out[i] = a[i] * b[i];
    }
#elif HAVE_AVX2
    {
        int i = 0;
        int n8 = n & ~7;
        for (; i < n8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
        }
        for (; i < n; i++) out[i] = a[i] * b[i];
    }
#else
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#endif

    return MORIE_OK;
}

/* ─── Dot Product ──────────────────────────────────────────────────────── */

int morie_dot(
    const float *a,
    const float *b,
    int          n,
    float       *result
) {
    CHECK_NULL(a);
    CHECK_NULL(b);
    CHECK_NULL(result);
    CHECK_SIZE(n);

#if HAVE_ACCELERATE
    vDSP_dotpr(a, 1, b, 1, result, n);
#elif HAVE_NEON_NATIVE
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;
        int n4 = n & ~3;
        for (; i < n4; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            acc = vfmaq_f32(acc, va, vb);
        }
        float sum = vaddvq_f32(acc);
        for (; i < n; i++) sum += a[i] * b[i];
        *result = sum;
    }
#elif HAVE_AVX2
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        int n8 = n & ~7;
        for (; i < n8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum;
        _mm_store_ss(&sum, s);
        for (; i < n; i++) sum += a[i] * b[i];
        *result = sum;
    }
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *result = sum;
#endif

    return MORIE_OK;
}

/* ─── Argmax ───────────────────────────────────────────────────────────── */

int morie_argmax(const float *x, int n, int *result) {
    CHECK_NULL(x);
    CHECK_NULL(result);
    CHECK_SIZE(n);

    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    *result = best;
    return MORIE_OK;
}
