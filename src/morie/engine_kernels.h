/*
 * engine_kernels.h — MORIE Inference Engine C Kernels
 *
 * Secure, portable C kernels for transformer forward pass operations.
 * Uses Accelerate.framework (vDSP/BLAS) on macOS for SIMD acceleration,
 * falls back to portable C on other platforms.
 *
 * Security:
 *   - All functions validate input sizes (no buffer overflows)
 *   - No heap allocation in hot-path functions (caller-allocated buffers)
 *   - No global mutable state (thread-safe)
 *   - No eval(), exec(), system(), or shell calls
 *
 * Compile:
 *   macOS:  cc -O2 -march=native -shared -o engine_kernels.dylib engine_kernels.c -lm -framework Accelerate
 *   Linux:  cc -O2 -march=native -shared -fPIC -o engine_kernels.so engine_kernels.c -lm -lopenblas
 */

#ifndef MORIE_ENGINE_KERNELS_H
#define MORIE_ENGINE_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes */
#define MORIE_OK          0
#define MORIE_ERR_NULL   -1   /* NULL pointer argument */
#define MORIE_ERR_SIZE   -2   /* Invalid size (0 or mismatched) */
#define MORIE_ERR_RANGE  -3   /* Index out of range */

/*
 * RMS Normalization: out[i] = x[i] * weight[i] / rms(x)
 *
 * x:       input vector (n elements)
 * weight:  learned scale (n elements)
 * out:     output vector (n elements, caller-allocated)
 * n:       vector length
 * eps:     epsilon for numerical stability (typically 1e-5)
 *
 * Returns: MORIE_OK on success, error code otherwise.
 */
int morie_rmsnorm(
    const float *x,
    const float *weight,
    float       *out,
    int          n,
    float        eps
);

/*
 * Rotary Position Embedding (RoPE): apply to query and key vectors.
 *
 * q:        query vector (head_dim elements, modified in-place)
 * k:        key vector (head_dim elements, modified in-place)
 * head_dim: dimension per head (must be even)
 * position: sequence position
 * freq_base: RoPE frequency base (typically 10000.0)
 *
 * Returns: MORIE_OK on success.
 */
int morie_rope(
    float *q,
    float *k,
    int    head_dim,
    int    position,
    float  freq_base
);

/*
 * Matrix-vector multiply: out = A @ x
 * Uses Accelerate.framework cblas_sgemv on macOS.
 *
 * A:    matrix (rows x cols, row-major)
 * x:    input vector (cols elements)
 * out:  output vector (rows elements, caller-allocated)
 * rows: number of rows in A
 * cols: number of columns in A
 *
 * Returns: MORIE_OK on success.
 */
int morie_matvec(
    const float *A,
    const float *x,
    float       *out,
    int          rows,
    int          cols
);

/*
 * SiLU activation in-place: x[i] = x[i] * sigmoid(x[i])
 *
 * x: vector (n elements, modified in-place)
 * n: vector length
 *
 * Returns: MORIE_OK on success.
 */
int morie_silu_inplace(float *x, int n);

/*
 * Softmax over a vector (in-place): x[i] = exp(x[i]) / sum(exp(x[j]))
 *
 * x: vector (n elements, modified in-place)
 * n: vector length
 *
 * Returns: MORIE_OK on success.
 */
int morie_softmax_inplace(float *x, int n);

/*
 * Elementwise multiply: out[i] = a[i] * b[i]
 *
 * a, b: input vectors (n elements each)
 * out:  output vector (n elements, caller-allocated; may alias a or b)
 * n:    vector length
 *
 * Returns: MORIE_OK on success.
 */
int morie_elemwise_mul(
    const float *a,
    const float *b,
    float       *out,
    int          n
);

/*
 * Dot product: return sum(a[i] * b[i])
 *
 * a, b: input vectors (n elements each)
 * n:    vector length
 * result: pointer to output scalar
 *
 * Returns: MORIE_OK on success.
 */
int morie_dot(
    const float *a,
    const float *b,
    int          n,
    float       *result
);

/*
 * Argmax: return index of maximum element.
 *
 * x: vector (n elements)
 * n: vector length
 * result: pointer to output index
 *
 * Returns: MORIE_OK on success.
 */
int morie_argmax(const float *x, int n, int *result);

#ifdef __cplusplus
}
#endif

#endif /* MORIE_ENGINE_KERNELS_H */
