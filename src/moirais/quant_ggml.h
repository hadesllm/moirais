/*
 * quant_ggml.h — TurboQuant block types for GGML compatibility
 *
 * Defines block structures matching llama.cpp PR #21089 (TBQ3_0, TBQ4_0)
 * for integration with the GGUF model format used by Ollama.
 *
 * References:
 *   - TurboQuant: arxiv.org/abs/2504.19874 (ICLR 2026)
 *   - PolarQuant: arxiv.org/abs/2502.02617 (AISTATS 2026)
 *   - llama.cpp PR: github.com/ggml-org/llama.cpp/pull/21089
 *
 * Block layout follows the TurboQuant paper's MSE-optimal scheme:
 *   1. Random rotation (Hadamard-Walsh or QR) — stored as seed
 *   2. Lloyd-Max quantization of rotated coordinates
 *   3. Bit-packed indices + L2 norm per block
 */

#ifndef MOIRAIS_QUANT_GGML_H
#define MOIRAIS_QUANT_GGML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Block sizes -------------------------------------------------------- */

/* Elements per quantization block */
#define TQ_BLOCK_SIZE 256

/* --- TBQ3_0: 3-bit TurboQuant (98 bytes per 256 elements) -------------- */
/*
 * Compression: 256 × 16 bits (FP16) = 512 bytes → 98 bytes = 5.2× ratio
 *
 * Layout:
 *   - norm:    float32  (4 bytes)  — L2 norm of the block
 *   - seed:    uint16   (2 bytes)  — rotation matrix seed
 *   - indices: 3 bits × 256 = 768 bits = 96 bytes (bit-packed)
 *                                   — Lloyd-Max codebook indices
 * Total: 4 + 2 + 96 = 102 bytes (paper spec: 98 with tighter packing)
 */
typedef struct __attribute__((packed)) {
    float    norm;                     /* L2 norm of original block      */
    uint16_t seed;                     /* rotation matrix seed           */
    uint8_t  indices[96];              /* bit-packed 3-bit indices       */
} block_tbq3_0;

_Static_assert(sizeof(block_tbq3_0) == 102,
               "block_tbq3_0 must be 102 bytes");

/* --- TBQ4_0: 4-bit TurboQuant (134 bytes per 256 elements) ------------- */

typedef struct __attribute__((packed)) {
    float    norm;                     /* L2 norm of original block      */
    uint16_t seed;                     /* rotation matrix seed           */
    uint8_t  indices[128];             /* bit-packed 4-bit indices       */
} block_tbq4_0;

_Static_assert(sizeof(block_tbq4_0) == 134,
               "block_tbq4_0 must be 134 bytes");

/* --- TBQ2_0: 2-bit TurboQuant (70 bytes per 256 elements) -------------- */

typedef struct __attribute__((packed)) {
    float    norm;
    uint16_t seed;
    uint8_t  indices[64];
} block_tbq2_0;

_Static_assert(sizeof(block_tbq2_0) == 70,
               "block_tbq2_0 must be 70 bytes");

/* --- Codebook ----------------------------------------------------------- */

/* Maximum bits supported */
#define TQ_MAX_BITS 4

/* Maximum codebook entries = 2^TQ_MAX_BITS */
#define TQ_MAX_CODEBOOK (1 << TQ_MAX_BITS)

/*
 * Pre-computed Lloyd-Max codebooks for common dimensions.
 * These are indexed as codebook[bits-1][k] for k in [0, 2^bits).
 *
 * For d=128 (typical transformer head_dim):
 *   2-bit: { -0.133, -0.040, 0.040, 0.133 }
 *   3-bit: { -0.189, -0.119, -0.067, -0.022, 0.022, 0.067, 0.119, 0.189 }
 *   4-bit: 16 centroids (computed at init)
 */
typedef struct {
    int     dim;                       /* embedding dimension            */
    int     bits;                      /* quantization bits              */
    int     n_centroids;               /* 2^bits                         */
    float   centroids[TQ_MAX_CODEBOOK]; /* sorted centroid values        */
} tq_codebook;

/* --- Quantization context ----------------------------------------------- */

typedef struct {
    int     dim;                       /* block dimension (256)          */
    int     bits;                      /* quantization bits (2-4)        */
    tq_codebook codebook;              /* Lloyd-Max codebook             */
    /* Rotation: use Hadamard-Walsh transform (no matrix storage needed) */
} tq_context;

/* --- Function declarations ---------------------------------------------- */

/*
 * Initialize a quantization context for the given dimension and bit-width.
 * Computes the Lloyd-Max codebook via iterative optimization.
 */
void tq_init(tq_context *ctx, int dim, int bits);

/*
 * Quantize a float vector into a TBQ block.
 *
 * src:     input vector of length ctx->dim (float32)
 * dst:     output block (caller-allocated)
 * ctx:     quantization context
 * seed:    rotation matrix seed
 */
void tq_quantize_block(
    const float *src,
    void        *dst,
    const tq_context *ctx,
    uint16_t seed
);

/*
 * Dequantize a TBQ block back to a float vector.
 *
 * src:     input block
 * dst:     output vector of length ctx->dim (float32, caller-allocated)
 * ctx:     quantization context
 */
void tq_dequantize_block(
    const void  *src,
    float       *dst,
    const tq_context *ctx
);

/*
 * Compute the dot product of a quantized vector with a float vector
 * WITHOUT full dequantization (fused kernel for attention).
 *
 * block:   quantized TBQ block
 * vec:     float vector of length ctx->dim
 * ctx:     quantization context
 *
 * Returns: approximate dot product
 */
float tq_dot_product(
    const void       *block,
    const float      *vec,
    const tq_context *ctx
);

/* --- QJL Stage 2: 1-bit error correction (Zandieh et al. AAAI 2025) ----- */
/* Reference: arXiv:2406.03482, github.com/amirzandieh/QJL                  */

typedef struct {
    int      dim;
    float   *projection;    /* d x d random N(0,1) matrix, row-major */
    uint32_t seed;
} tq_qjl_context;

void tq_qjl_init(tq_qjl_context *ctx, int dim, uint32_t seed);
void tq_qjl_free(tq_qjl_context *ctx);

void tq_qjl_encode(
    const float *residual, uint8_t *signs, float *norm_out,
    const tq_qjl_context *ctx
);

void tq_qjl_decode(
    const uint8_t *signs, float norm, float *dst,
    const tq_qjl_context *ctx
);

/* Full two-stage pipeline: Stage 1 (MSE) + Stage 2 (QJL) */
void tq_quantize_block_prod(
    const float *src, void *dst_stage1,
    uint8_t *qjl_signs, float *qjl_norm,
    const tq_context *ctx, const tq_qjl_context *qjl_ctx,
    uint16_t seed
);

void tq_dequantize_block_prod(
    const void *stage1_block, const uint8_t *qjl_signs, float qjl_norm,
    float *dst,
    const tq_context *ctx, const tq_qjl_context *qjl_ctx
);

#ifdef __cplusplus
}
#endif

#endif /* MOIRAIS_QUANT_GGML_H */
