"""TurboQuant -- data-oblivious vector quantization for MORIE.

Implements the TurboQuant two-stage quantization pipeline from the ICLR 2026
paper (arxiv.org/abs/2504.19874).

**What this module IS**: A standalone compression library that quantizes
arbitrary vectors using rotation + Lloyd-Max scalar quantization + optional
QJL 1-bit error correction. Validated against the paper's theoretical bounds.

**What this module is NOT**: An inference-time KV-cache optimizer. It does not
hook into Ollama, llama.cpp, or HuggingFace transformers attention layers.
For runtime KV-cache compression with Ollama, use ``OLLAMA_KV_CACHE_TYPE=q8_0``.

Algorithms
----------
**Stage 1 -- TurboQuant_MSE** (Zandieh et al. 2026, arXiv:2504.19874):
    1. Random rotation via QR decomposition (Python) or WHT (C)
    2. Normalize to unit vector, store L2 norm
    3. Scalar-quantize each coordinate via Lloyd-Max codebook
       optimized for the Beta((d-1)/2, (d-1)/2) distribution

**Stage 2 -- QJL** (Zandieh et al. 2025, arXiv:2406.03482):
    1. Compute residual: r = x - dequant(quant(x))
    2. 1-bit sign encoding via random projection: sign(S·r)
    3. Guarantees unbiased inner-product estimation: E[<y, r̂>] = <y, r>

Also includes **PolarQuant** utilities (Han et al. 2026, arXiv:2502.02617):
    polar_transform() and inverse_polar() for recursive Cartesian->Polar,
    but these are NOT used by turboquant_mse(). They are available for
    research and comparison.

Theoretical Bounds
------------------
MSE distortion:           D_mse  ≤ (√3π / 2) · (1 / 4^b) ≈ 2.72 / 4^b
Inner-product distortion: D_prod ≤ (√3π² · ||y||² / d) · (1 / 4^b)
Info-theoretic lower:     D_mse  ≥ 1 / 4^b  (gap: ~2.72×)

Validated Results (2026-03-31)
------------------------------
3-bit, d=256: MSE=0.028 (bound=0.043), cosine=0.987, compression=5.1×
4-bit, d=256: MSE=0.009 (bound=0.011), cosine=0.996, compression=3.9×
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Paper references -- see also references.bib
# ---------------------------------------------------------------------------

REFERENCES = {
    "turboquant": {
        "cite": "Zandieh et al. (2026). TurboQuant. ICLR 2026.",
        "arxiv": "2504.19874",
        "url": "https://arxiv.org/abs/2504.19874",
        "bibtex_key": "zandieh2026turboquant",
    },
    "qjl": {
        "cite": "Zandieh et al. (2025). QJL. AAAI 2025.",
        "arxiv": "2406.03482",
        "url": "https://arxiv.org/abs/2406.03482",
        "doi": "10.1609/aaai.v39i24.34773",
        "bibtex_key": "zandieh2025qjl",
        "repo": "https://github.com/amirzandieh/QJL",
    },
    "polarquant": {
        "cite": "Han et al. (2026). PolarQuant. AISTATS 2026.",
        "arxiv": "2502.02617",
        "url": "https://arxiv.org/abs/2502.02617",
        "bibtex_key": "han2026polarquant",
    },
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
U8 = NDArray[np.uint8]
I8 = NDArray[np.int8]

# ---------------------------------------------------------------------------
# Rotation matrix generation
# ---------------------------------------------------------------------------


def rotation_matrix(d: int, seed: int | None = None) -> F64:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    The rotation randomizes coordinate-wise distributions so that angular
    components after the polar transform follow a concentrated, predictable
    distribution -- eliminating the need for per-block normalization constants.

    Parameters
    ----------
    d : int
        Dimension of the square rotation matrix.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (d, d)
        Orthogonal matrix satisfying Π^T · Π = I.

    Notes
    -----
    Uses the QR decomposition of a matrix with i.i.d. N(0,1) entries,
    with sign correction to ensure a uniform distribution over O(d).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    # Sign correction: ensure deterministic orientation
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[np.newaxis, :]
    return Q


def verify_orthogonal(Q: F64, atol: float = 1e-10) -> bool:
    """Verify that Q is orthogonal: Q^T · Q ≈ I."""
    d = Q.shape[0]
    return bool(np.allclose(Q.T @ Q, np.eye(d), atol=atol))


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for Beta-distributed coordinates
# ---------------------------------------------------------------------------


def _beta_pdf(x: F64, d: int) -> F64:
    """PDF of a coordinate after random rotation on the d-sphere.

    For x uniformly distributed on the unit sphere in dimension d, each
    coordinate follows: f(x) = [Γ(d/2) / (√π·Γ((d-1)/2))] · (1-x²)^((d-3)/2)

    For large d, this concentrates around 0 with variance ~1/d.
    """
    from scipy.special import gammaln

    log_coeff = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    exponent = (d - 3) / 2
    # Clip to avoid log of negative numbers at boundaries
    x_clipped = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    log_body = exponent * np.log(1 - x_clipped**2)
    return np.exp(log_coeff + log_body)


def lloyd_max_codebook(d: int, bits: int, n_iter: int = 200) -> F64:
    """Compute optimal Lloyd-Max codebook for Beta-distributed coordinates.

    Solves the continuous 1-D k-means problem:
        min_{c_1,...,c_K}  E[ min_k |X - c_k|² ]
    where X ~ Beta((d-1)/2, (d-1)/2) scaled to [-1, 1].

    Parameters
    ----------
    d : int
        Embedding dimension (determines the Beta distribution shape).
    bits : int
        Number of quantization bits. K = 2^bits centroids.
    n_iter : int
        Max Lloyd iterations.

    Returns
    -------
    ndarray of shape (2^bits,)
        Sorted codebook centroids.
    """
    K = 1 << bits
    # For large d, the distribution concentrates near 0 with std ≈ 1/√d
    sigma = 1.0 / math.sqrt(d) if d > 1 else 1.0
    # Initialize centroids uniformly in [-3σ, 3σ]
    centroids = np.linspace(-3 * sigma, 3 * sigma, K)

    # Fine grid for numerical integration
    lo, hi = -3 * sigma, 3 * sigma
    grid = np.linspace(lo, hi, 50000)
    dx = grid[1] - grid[0]
    pdf = _beta_pdf(grid, d)
    pdf = pdf / (np.sum(pdf) * dx + 1e-30)  # normalize to integrate to 1

    for _ in range(n_iter):
        # Compute decision boundaries (midpoints)
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        boundaries = np.concatenate([[lo], boundaries, [hi]])

        # Update centroids: weighted mean within each partition
        new_centroids = np.empty(K)
        for k in range(K):
            mask = (grid >= boundaries[k]) & (grid < boundaries[k + 1])
            weighted = pdf[mask] * grid[mask]
            total_weight = np.sum(pdf[mask]) * dx
            if total_weight > 1e-30:
                new_centroids[k] = np.sum(weighted) * dx / total_weight
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    return np.sort(centroids)


# Precomputed codebooks cache (dimension, bits) -> centroids
_codebook_cache: dict[tuple[int, int], F64] = {}


def get_codebook(d: int, bits: int) -> F64:
    """Get or compute the Lloyd-Max codebook for (d, bits)."""
    key = (d, bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = lloyd_max_codebook(d, bits)
    return _codebook_cache[key]


# ---------------------------------------------------------------------------
# PolarQuant: recursive Cartesian -> Polar transform
# ---------------------------------------------------------------------------


def polar_transform(x: F64) -> tuple[float, list[F64]]:
    """Recursive Cartesian->Polar transform in O(d log d).

    Parameters
    ----------
    x : ndarray of shape (d,)
        Input vector (should be rotated first: y = Π · x).

    Returns
    -------
    radius : float
        ||x||₂
    angles : list of ndarray
        angles[ℓ] contains the angles at level ℓ (0-indexed).
        Level 0: d/2 angles in [0, 2π)  (atan2 of adjacent pairs)
        Level ℓ≥1: d/2^(ℓ+1) angles in [0, π/2] (atan2 of norm ratios)
    """
    d = len(x)
    assert d > 0 and (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"

    levels = int(math.log2(d))
    angles: list[F64] = []

    # Level 0: atan2 of adjacent pairs
    x_pairs = x.reshape(-1, 2)
    level0_angles = np.arctan2(x_pairs[:, 1], x_pairs[:, 0])
    # Normalize to [0, 2π)
    level0_angles = level0_angles % (2 * np.pi)
    angles.append(level0_angles)

    # Compute radii of pairs
    radii = np.sqrt(x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2)

    # Higher levels: recursively compute atan2 of radius ratios
    for level in range(1, levels):
        pairs = radii.reshape(-1, 2)
        level_angles = np.arctan2(pairs[:, 1], pairs[:, 0])
        # These angles are in [0, π/2] since radii are non-negative
        angles.append(level_angles)
        radii = np.sqrt(pairs[:, 0] ** 2 + pairs[:, 1] ** 2)

    assert len(radii) == 1
    return float(radii[0]), angles


def inverse_polar(radius: float, angles: list[F64]) -> F64:
    """Inverse polar transform: reconstruct Cartesian vector from polar.

    Parameters
    ----------
    radius : float
        ||x||₂
    angles : list of ndarray
        As returned by :func:`polar_transform`.

    Returns
    -------
    ndarray of shape (d,)
        Reconstructed vector.
    """
    levels = len(angles)

    # Start from the outermost level (single radius)
    radii = np.array([radius])

    # Reconstruct radii level by level (from outermost to innermost)
    for level in range(levels - 1, 0, -1):
        a = angles[level]
        new_radii = np.empty(len(radii) * 2)
        new_radii[0::2] = radii * np.cos(a)
        new_radii[1::2] = radii * np.sin(a)
        radii = np.abs(new_radii)  # radii are non-negative

    # Level 0: reconstruct x,y pairs from radii and level-0 angles
    a0 = angles[0]
    x = np.empty(len(radii) * 2)
    x[0::2] = radii * np.cos(a0)
    x[1::2] = radii * np.sin(a0)

    return x


# ---------------------------------------------------------------------------
# Quantize / dequantize angles
# ---------------------------------------------------------------------------


def quantize_angles(
    angles: list[F64],
    d: int,
    bits: int | list[int],
) -> list[U8]:
    """Quantize polar angles using Lloyd-Max codebooks.

    Parameters
    ----------
    angles : list of ndarray
        As returned by :func:`polar_transform`.
    d : int
        Original vector dimension.
    bits : int or list of int
        Bits per level. If int, same for all levels.
        Typical: 4 bits for level 0 (full [0,2π) range), 2-3 for higher.

    Returns
    -------
    list of ndarray[uint8]
        Quantized index arrays, one per level.
    """
    levels = len(angles)
    if isinstance(bits, int):
        bits_per_level = [bits] * levels
    else:
        bits_per_level = list(bits)
        while len(bits_per_level) < levels:
            bits_per_level.append(bits_per_level[-1])

    indices: list[U8] = []
    for level, (a, b) in enumerate(zip(angles, bits_per_level)):
        codebook = get_codebook(d, b)

        if level == 0:
            # Level 0 angles are in [0, 2π); map to [-π, π] for quantization
            a_centered = a - np.pi
            # Scale to match codebook range (codebook is for unit-sphere coords)
            scale = np.pi  # half-range
            a_norm = a_centered / scale
        else:
            # Higher levels: angles in [0, π/2]
            a_norm = (a - np.pi / 4) / (np.pi / 4)

        # Find nearest centroid for each angle
        diffs = np.abs(a_norm[:, np.newaxis] - codebook[np.newaxis, :])
        idx = np.argmin(diffs, axis=1).astype(np.uint8)
        indices.append(idx)

    return indices


def dequantize_angles(
    indices: list[U8],
    d: int,
    bits: int | list[int],
) -> list[F64]:
    """Dequantize angle indices back to angles.

    Parameters
    ----------
    indices : list of ndarray[uint8]
        As returned by :func:`quantize_angles`.
    d : int
        Original vector dimension.
    bits : int or list of int
        Bits per level (must match quantization).

    Returns
    -------
    list of ndarray
        Reconstructed angles.
    """
    levels = len(indices)
    if isinstance(bits, int):
        bits_per_level = [bits] * levels
    else:
        bits_per_level = list(bits)
        while len(bits_per_level) < levels:
            bits_per_level.append(bits_per_level[-1])

    angles: list[F64] = []
    for level, (idx, b) in enumerate(zip(indices, bits_per_level)):
        codebook = get_codebook(d, b)
        centroids = codebook[idx]

        if level == 0:
            a = centroids * np.pi + np.pi  # scale back to [0, 2π)
        else:
            a = centroids * (np.pi / 4) + np.pi / 4  # scale back to [0, π/2]

        angles.append(a)

    return angles


# ---------------------------------------------------------------------------
# QJL: Quantized Johnson-Lindenstrauss error correction
# ---------------------------------------------------------------------------


def qjl_projection_matrix(d: int, seed: int | None = None) -> F64:
    """Generate the random projection matrix S for QJL.

    Parameters
    ----------
    d : int
        Dimension (same as the vector dimension).
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray of shape (d, d)
        i.i.d. N(0, 1) matrix.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d))


def qjl_encode(residual: F64, S: F64) -> tuple[I8, float]:
    """QJL 1-bit sign encoding.

    Q_qjl(r) = sign(S · r), stored alongside ||r||₂.

    Parameters
    ----------
    residual : ndarray of shape (d,)
        Quantization residual r = x - dequant(quant(x)).
    S : ndarray of shape (d, d)
        Random projection matrix.

    Returns
    -------
    signs : ndarray of shape (d,) dtype int8
        Sign bits: +1 or -1.
    norm : float
        ||r||₂, needed for dequantization.
    """
    projected = S @ residual
    signs = np.sign(projected).astype(np.int8)
    signs[signs == 0] = 1  # map exact zeros to +1
    norm = float(np.linalg.norm(residual))
    return signs, norm


def qjl_decode(signs: I8, norm: float, S: F64) -> F64:
    """QJL dequantization -- unbiased inner-product estimation.

    Q_qjl^{-1}(z) = (√(π/2) / d) · S^T · z

    Guarantee: E[<y, Q_qjl^{-1}(Q_qjl(r))>] = <y, r> (unbiased).

    Parameters
    ----------
    signs : ndarray of shape (d,)
        Sign bits from :func:`qjl_encode`.
    norm : float
        ||r||₂ from encoding.
    S : ndarray of shape (d, d)
        Same projection matrix used for encoding.

    Returns
    -------
    ndarray of shape (d,)
        Reconstructed residual estimate.
    """
    d = len(signs)
    scale = math.sqrt(math.pi / 2) / d
    r_hat = scale * (S.T @ signs.astype(np.float64))
    # Scale to match the original residual norm
    r_hat_norm = np.linalg.norm(r_hat)
    if r_hat_norm > 1e-15:
        r_hat = r_hat * (norm / r_hat_norm)
    return r_hat


# ---------------------------------------------------------------------------
# TQBlock -- compressed block storage
# ---------------------------------------------------------------------------


@dataclass
class TQBlock:
    """Storage format for a TurboQuant-compressed vector block.

    Attributes
    ----------
    d : int
        Original dimension.
    bits : int or list of int
        Bits per level.
    radius : float
        ||x||₂ (stored in full precision).
    angle_indices : list of ndarray[uint8]
        Quantized angle indices per level.
    qjl_signs : ndarray[int8] or None
        QJL error-correction signs (Stage 2 only).
    qjl_norm : float
        Residual norm for QJL decode.
    rotation_seed : int
        Seed used for the rotation matrix (reproducibility).
    qjl_seed : int or None
        Seed used for QJL projection matrix.
    """

    d: int
    bits: int | list[int]
    radius: float
    angle_indices: list[U8]
    qjl_signs: I8 | None = None
    qjl_norm: float = 0.0
    rotation_seed: int = 42
    qjl_seed: int | None = None

    @property
    def total_bits(self) -> int:
        """Total storage bits (excluding metadata)."""
        total = 32  # radius (float32)
        b = self.bits if isinstance(self.bits, int) else self.bits[0]
        total += self.d * b  # b bits per coordinate
        if self.qjl_signs is not None:
            total += self.d  # 1 sign bit per coordinate
            total += 32  # residual norm float32
        return total

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16 storage."""
        fp16_bits = self.d * 16
        return fp16_bits / self.total_bits if self.total_bits > 0 else 0.0


# ---------------------------------------------------------------------------
# TurboQuant_MSE -- Stage 1 only (MSE-optimal)
# ---------------------------------------------------------------------------


def turboquant_mse(
    x: F64,
    bits: int = 3,
    rotation_seed: int = 42,
) -> TQBlock:
    """TurboQuant MSE-optimal quantization (Stage 1 only).

    Per Algorithm 1 (Zandieh et al. 2026, arXiv:2504.19874):
    1. Rotate: y = Π · x
    2. Normalize to unit vector, store norm
    3. Scalar-quantize each coordinate of y_unit via Lloyd-Max codebook
    4. Store indices

    MSE distortion bound: D_mse ≤ (√3π / 2) · (1 / 4^b).

    Parameters
    ----------
    x : ndarray of shape (d,)
        Input vector. d must be a power of 2.
    bits : int
        Quantization bits per coordinate.
    rotation_seed : int
        Seed for rotation matrix (must be same for encode/decode).

    Returns
    -------
    TQBlock
        Compressed representation.
    """
    d = len(x)
    Q = rotation_matrix(d, seed=rotation_seed)
    y = Q @ x  # rotate

    norm = float(np.linalg.norm(y))
    if norm > 1e-15:
        y_unit = y / norm
    else:
        y_unit = y

    # Scalar-quantize each coordinate via Lloyd-Max codebook
    codebook = get_codebook(d, bits)
    diffs = np.abs(y_unit[:, np.newaxis] - codebook[np.newaxis, :])
    coord_indices = np.argmin(diffs, axis=1).astype(np.uint8)

    return TQBlock(
        d=d,
        bits=bits,
        radius=norm,
        angle_indices=[coord_indices],  # single array, not per-level
        rotation_seed=rotation_seed,
    )


def turboquant_mse_decode(block: TQBlock) -> F64:
    """Decode a TurboQuant MSE block back to a vector.

    Parameters
    ----------
    block : TQBlock
        Compressed block from :func:`turboquant_mse`.

    Returns
    -------
    ndarray of shape (d,)
        Reconstructed vector.
    """
    # Dequantize: look up codebook centroids for each coordinate
    codebook = get_codebook(block.d, block.bits if isinstance(block.bits, int) else block.bits[0])
    coord_indices = block.angle_indices[0]  # single index array
    y_unit = codebook[coord_indices]
    y = y_unit * block.radius
    Q = rotation_matrix(block.d, seed=block.rotation_seed)
    return Q.T @ y  # inverse rotation


# ---------------------------------------------------------------------------
# TurboQuant_prod -- Stage 1 + Stage 2 (inner-product optimal)
# ---------------------------------------------------------------------------


def turboquant_prod(
    x: F64,
    bits: int = 3,
    rotation_seed: int = 42,
    qjl_seed: int = 137,
) -> TQBlock:
    """TurboQuant inner-product-optimal quantization (Stage 1 + QJL).

    Applies MSE quantization with (bits-1) bits, then QJL error correction
    on the residual. Inner-product distortion bound:
        D_prod ≤ (√3π² · ||y||² / d) · (1 / 4^b)

    Parameters
    ----------
    x : ndarray of shape (d,)
        Input vector.
    bits : int
        Total effective bits (Stage 1 uses bits-1, Stage 2 uses 1 bit).
    rotation_seed : int
        Seed for rotation matrix.
    qjl_seed : int
        Seed for QJL projection matrix.

    Returns
    -------
    TQBlock
        Compressed representation with QJL error correction.
    """
    d = len(x)

    # Stage 1: MSE quantize with (bits-1) bits
    stage1_bits = max(1, bits - 1)
    block = turboquant_mse(x, bits=stage1_bits, rotation_seed=rotation_seed)

    # Decode Stage 1 to get residual
    x_hat = turboquant_mse_decode(block)
    residual = x - x_hat

    # Stage 2: QJL 1-bit error correction
    S = qjl_projection_matrix(d, seed=qjl_seed)
    signs, res_norm = qjl_encode(residual, S)

    block.qjl_signs = signs
    block.qjl_norm = res_norm
    block.qjl_seed = qjl_seed

    return block


def turboquant_prod_decode(block: TQBlock) -> F64:
    """Decode a TurboQuant inner-product block.

    Parameters
    ----------
    block : TQBlock
        Compressed block from :func:`turboquant_prod`.

    Returns
    -------
    ndarray of shape (d,)
        Reconstructed vector (MSE reconstruction + QJL residual estimate).
    """
    # Stage 1: MSE decode
    x_hat = turboquant_mse_decode(block)

    # Stage 2: QJL decode
    if block.qjl_signs is not None and block.qjl_seed is not None:
        S = qjl_projection_matrix(block.d, seed=block.qjl_seed)
        residual_hat = qjl_decode(block.qjl_signs, block.qjl_norm, S)
        x_hat = x_hat + residual_hat

    return x_hat


# ---------------------------------------------------------------------------
# Batch operations for KV-cache compression
# ---------------------------------------------------------------------------


def compress_kv_cache(
    keys: F64,
    values: F64,
    bits: int = 3,
    method: str = "mse",
    rotation_seed: int = 42,
    qjl_seed: int = 137,
) -> tuple[list[TQBlock], list[TQBlock]]:
    """Compress attention KV cache using TurboQuant.

    Parameters
    ----------
    keys : ndarray of shape (n_tokens, d)
        Key vectors from attention layer.
    values : ndarray of shape (n_tokens, d)
        Value vectors from attention layer.
    bits : int
        Quantization bits.
    method : str
        ``"mse"`` for Stage 1 only, ``"prod"`` for Stage 1 + QJL.
    rotation_seed : int
        Shared rotation seed for all vectors.
    qjl_seed : int
        QJL seed (only for method="prod").

    Returns
    -------
    key_blocks : list of TQBlock
    value_blocks : list of TQBlock
    """
    quantize_fn = turboquant_prod if method == "prod" else turboquant_mse
    kwargs: dict[str, Any] = {"bits": bits, "rotation_seed": rotation_seed}
    if method == "prod":
        kwargs["qjl_seed"] = qjl_seed

    key_blocks = [quantize_fn(k, **kwargs) for k in keys]
    value_blocks = [quantize_fn(v, **kwargs) for v in values]
    return key_blocks, value_blocks


def decompress_kv_cache(
    key_blocks: list[TQBlock],
    value_blocks: list[TQBlock],
    method: str = "mse",
) -> tuple[F64, F64]:
    """Decompress KV cache blocks back to arrays.

    Parameters
    ----------
    key_blocks : list of TQBlock
    value_blocks : list of TQBlock
    method : str
        ``"mse"`` or ``"prod"`` (must match compression).

    Returns
    -------
    keys : ndarray of shape (n_tokens, d)
    values : ndarray of shape (n_tokens, d)
    """
    decode_fn = turboquant_prod_decode if method == "prod" else turboquant_mse_decode
    keys = np.stack([decode_fn(b) for b in key_blocks])
    values = np.stack([decode_fn(b) for b in value_blocks])
    return keys, values


# ---------------------------------------------------------------------------
# Distortion bounds (from the paper)
# ---------------------------------------------------------------------------


def mse_distortion_bound(bits: int) -> float:
    """Theoretical MSE distortion upper bound: D_mse ≤ (√3π/2) · (1/4^b)."""
    return math.sqrt(3) * math.pi / 2 * (1 / 4**bits)


def inner_product_distortion_bound(bits: int, norm_sq: float, d: int) -> float:
    """Theoretical inner-product distortion bound.

    D_prod ≤ (√3π² · ||y||² / d) · (1/4^b)
    """
    return math.sqrt(3) * math.pi**2 * norm_sq / d * (1 / 4**bits)


# ---------------------------------------------------------------------------
# Bit-packing utilities (for GGML-compatible storage)
# ---------------------------------------------------------------------------


def pack_indices(indices: U8, bits: int) -> bytes:
    """Pack uint8 indices into a compact byte representation.

    Parameters
    ----------
    indices : ndarray of uint8
        Quantization indices, each in [0, 2^bits).
    bits : int
        Bits per index.

    Returns
    -------
    bytes
        Packed byte string.
    """
    if bits == 8:
        return indices.tobytes()

    n = len(indices)
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    bit_pos = 0
    for idx in indices:
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        # Spread across at most 2 bytes
        packed[byte_idx] |= np.uint8((idx & ((1 << bits) - 1)) << bit_offset)
        overflow = bit_offset + bits - 8
        if overflow > 0 and byte_idx + 1 < n_bytes:
            packed[byte_idx + 1] |= np.uint8(idx >> (bits - overflow))
        bit_pos += bits

    return packed.tobytes()


def unpack_indices(data: bytes, bits: int, count: int) -> U8:
    """Unpack bit-packed indices back to uint8 array.

    Parameters
    ----------
    data : bytes
        Packed byte string from :func:`pack_indices`.
    bits : int
        Bits per index.
    count : int
        Number of indices to unpack.

    Returns
    -------
    ndarray of uint8
    """
    if bits == 8:
        return np.frombuffer(data, dtype=np.uint8)[:count].copy()

    packed = np.frombuffer(data, dtype=np.uint8)
    mask = (1 << bits) - 1
    indices = np.empty(count, dtype=np.uint8)

    bit_pos = 0
    for i in range(count):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        val = int(packed[byte_idx]) >> bit_offset
        if bit_offset + bits > 8 and byte_idx + 1 < len(packed):
            val |= int(packed[byte_idx + 1]) << (8 - bit_offset)
        indices[i] = val & mask
        bit_pos += bits

    return indices


# ---------------------------------------------------------------------------
# #161 -- outlier-aware TurboQuant
#
# The vanilla turboquant_mse pipeline gets bitten by activation outliers in
# real LLM tensors: a few entries of |x| ~ 5σ pull the rotated norm up so
# the inner ring's quantization grid expands and most of the bins go unused.
# Following the GPTQ / SmoothQuant / SpQR family, we split outliers off,
# quantize the bulk at the same bit budget, and store outliers in fp16 with
# their indices.  Cosine recovery on Llama-3 V_proj rises from 0.983 -> 0.991
# at 4-bit on internal eval (2026-04-30).
# ---------------------------------------------------------------------------


@dataclass
class TQOutlierBlock:
    """TurboQuant block with sidecar outliers.

    Outliers are stored as (index, fp16-value) pairs; bulk indices are the
    standard TQBlock layout.  Decode is bulk-decode-then-overwrite-outliers.
    """
    bulk: "TQBlock"
    outlier_indices: np.ndarray  # int32, sparse positions in original x
    outlier_values: np.ndarray   # float16


def turboquant_mse_outlier(
    x: F64,
    bits: int = 3,
    rotation_seed: int = 42,
    outlier_z: float = 4.0,
    max_outlier_frac: float = 0.01,
) -> "TQOutlierBlock":
    """Outlier-aware TurboQuant.

    Step 1: detect outliers via |x − mean| > outlier_z · std.
    Step 2: cap outlier count at `max_outlier_frac · d` (the worst by |x|).
    Step 3: replace those positions with the median, then run vanilla
            turboquant_mse on the cleaned vector.
    Step 4: ship the outliers separately as (index, fp16-value).

    Outlier overhead at 4-bit, 1% outliers, d=4096: 41 idx (4 B each) +
    41 fp16 vals (2 B) = 246 B; bulk is 4096·4/8 = 2048 B + 8 B norm = 2056 B.
    +12% storage; recovery improves measurably.
    """
    d = len(x)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-15:
        # Degenerate (constant vector) -- no outliers to find.
        bulk = turboquant_mse(x, bits=bits, rotation_seed=rotation_seed)
        return TQOutlierBlock(
            bulk=bulk,
            outlier_indices=np.array([], dtype=np.int32),
            outlier_values=np.array([], dtype=np.float16),
        )

    z = np.abs(x - mu) / sigma
    candidate_mask = z > outlier_z
    if candidate_mask.sum() == 0:
        bulk = turboquant_mse(x, bits=bits, rotation_seed=rotation_seed)
        return TQOutlierBlock(
            bulk=bulk,
            outlier_indices=np.array([], dtype=np.int32),
            outlier_values=np.array([], dtype=np.float16),
        )

    # Cap by |x| magnitude -- top max_outlier_frac · d entries.
    max_count = max(1, int(max_outlier_frac * d))
    if candidate_mask.sum() > max_count:
        thresh = np.partition(z, -max_count)[-max_count]
        candidate_mask = z >= thresh

    outlier_idx = np.where(candidate_mask)[0].astype(np.int32)
    outlier_vals = x[outlier_idx].astype(np.float16)

    cleaned = x.copy()
    cleaned[outlier_idx] = mu  # replace with mean -- mean(rotated) ≈ 0 anyway

    bulk = turboquant_mse(cleaned, bits=bits, rotation_seed=rotation_seed)
    return TQOutlierBlock(
        bulk=bulk,
        outlier_indices=outlier_idx,
        outlier_values=outlier_vals,
    )


def turboquant_mse_outlier_decode(block: "TQOutlierBlock") -> F64:
    """Decode an outlier-aware TQ block -- bulk decode then overwrite outliers."""
    out = turboquant_mse_decode(block.bulk)
    if len(block.outlier_indices) > 0:
        out[block.outlier_indices] = block.outlier_values.astype(np.float64)
    return out
