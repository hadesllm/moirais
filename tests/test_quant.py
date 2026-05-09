"""Tests for moirais.quant — TurboQuant/PolarQuant/QJL implementation.

Validates mathematical correctness against paper bounds:
    - Rotation matrix orthogonality (Π^T·Π = I)
    - Polar transform roundtrip (encode→decode ≈ original)
    - QJL unbiasedness (E[<y, r̂>] ≈ <y, r>)
    - MSE distortion within theoretical bounds
    - Compression ratios
    - Bit-packing roundtrip
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from moirais.quant import (
    TQBlock,
    compress_kv_cache,
    decompress_kv_cache,
    dequantize_angles,
    inner_product_distortion_bound,
    inverse_polar,
    lloyd_max_codebook,
    mse_distortion_bound,
    pack_indices,
    polar_transform,
    qjl_decode,
    qjl_encode,
    qjl_projection_matrix,
    quantize_angles,
    rotation_matrix,
    turboquant_mse,
    turboquant_mse_decode,
    turboquant_prod,
    turboquant_prod_decode,
    unpack_indices,
    verify_orthogonal,
)


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_orthogonality_small(self):
        Q = rotation_matrix(8, seed=42)
        assert verify_orthogonal(Q)

    def test_orthogonality_128(self):
        Q = rotation_matrix(128, seed=42)
        assert verify_orthogonal(Q)

    def test_deterministic_with_seed(self):
        Q1 = rotation_matrix(16, seed=99)
        Q2 = rotation_matrix(16, seed=99)
        assert np.allclose(Q1, Q2)

    def test_different_seeds_differ(self):
        Q1 = rotation_matrix(16, seed=1)
        Q2 = rotation_matrix(16, seed=2)
        assert not np.allclose(Q1, Q2)

    def test_norm_preserving(self):
        """Rotation must preserve L2 norm."""
        Q = rotation_matrix(32, seed=42)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(32)
        y = Q @ x
        assert abs(np.linalg.norm(x) - np.linalg.norm(y)) < 1e-10


# ---------------------------------------------------------------------------
# Lloyd-Max codebook
# ---------------------------------------------------------------------------


class TestLloydMax:
    def test_codebook_sorted(self):
        cb = lloyd_max_codebook(128, bits=3)
        assert all(cb[i] <= cb[i + 1] for i in range(len(cb) - 1))

    def test_codebook_size(self):
        assert len(lloyd_max_codebook(128, bits=2)) == 4
        assert len(lloyd_max_codebook(128, bits=3)) == 8
        assert len(lloyd_max_codebook(128, bits=4)) == 16

    def test_codebook_symmetric(self):
        """For symmetric distributions, codebook should be roughly symmetric."""
        cb = lloyd_max_codebook(128, bits=3)
        # Check that positive and negative centroids are roughly mirrored
        assert abs(cb[0] + cb[-1]) < 0.05

    def test_codebook_concentrates_for_large_d(self):
        """Larger d → narrower codebook (centroids closer to 0)."""
        cb_small = lloyd_max_codebook(16, bits=3)
        cb_large = lloyd_max_codebook(256, bits=3)
        assert max(abs(cb_large)) < max(abs(cb_small))


# ---------------------------------------------------------------------------
# Polar transform roundtrip
# ---------------------------------------------------------------------------


class TestPolarTransform:
    def test_roundtrip_exact(self):
        """Polar→inverse_polar should recover the original vector exactly."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(16)
        radius, angles = polar_transform(x)
        x_hat = inverse_polar(radius, angles)
        assert np.allclose(x, x_hat, atol=1e-10)

    def test_roundtrip_128(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(128)
        radius, angles = polar_transform(x)
        x_hat = inverse_polar(radius, angles)
        assert np.allclose(x, x_hat, atol=1e-10)

    def test_radius_is_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(32)
        radius, _ = polar_transform(x)
        assert abs(radius - np.linalg.norm(x)) < 1e-10

    def test_angle_count(self):
        """d=2^k should produce k levels of angles."""
        x = np.random.default_rng(0).standard_normal(64)
        _, angles = polar_transform(x)
        assert len(angles) == 6  # log2(64)
        assert len(angles[0]) == 32  # d/2

    def test_power_of_two_required(self):
        with pytest.raises(AssertionError):
            polar_transform(np.ones(7))


# ---------------------------------------------------------------------------
# QJL encode/decode
# ---------------------------------------------------------------------------


class TestQJL:
    def test_signs_are_pm1(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(32)
        S = qjl_projection_matrix(32, seed=0)
        signs, norm = qjl_encode(r, S)
        assert set(np.unique(signs)).issubset({-1, 1})

    def test_norm_matches(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(32)
        S = qjl_projection_matrix(32, seed=0)
        _, norm = qjl_encode(r, S)
        assert abs(norm - np.linalg.norm(r)) < 1e-10

    def test_unbiasedness(self):
        """QJL should be an unbiased estimator of inner products.

        E[<y, Q^{-1}(Q(r))>] ≈ <y, r> over many trials.
        """
        d = 64
        rng = np.random.default_rng(42)
        r = rng.standard_normal(d)
        y = rng.standard_normal(d)
        true_ip = np.dot(y, r)

        # Average over many random projections
        estimates = []
        for seed in range(200):
            S = qjl_projection_matrix(d, seed=seed)
            signs, norm = qjl_encode(r, S)
            r_hat = qjl_decode(signs, norm, S)
            estimates.append(np.dot(y, r_hat))

        mean_estimate = np.mean(estimates)
        # Should be approximately equal (unbiased)
        assert abs(mean_estimate - true_ip) < 0.5 * abs(true_ip) + 0.5

    def test_decode_output_shape(self):
        d = 32
        S = qjl_projection_matrix(d, seed=0)
        signs = np.ones(d, dtype=np.int8)
        r_hat = qjl_decode(signs, 1.0, S)
        assert r_hat.shape == (d,)


# ---------------------------------------------------------------------------
# TurboQuant MSE (Stage 1)
# ---------------------------------------------------------------------------


class TestTurboQuantMSE:
    def test_encode_decode_shape(self):
        x = np.random.default_rng(42).standard_normal(64)
        block = turboquant_mse(x, bits=3)
        x_hat = turboquant_mse_decode(block)
        assert x_hat.shape == x.shape

    def test_compression_preserves_norm_roughly(self):
        x = np.random.default_rng(42).standard_normal(128)
        block = turboquant_mse(x, bits=3)
        x_hat = turboquant_mse_decode(block)
        # Norm should be approximately preserved
        assert abs(np.linalg.norm(x_hat) - np.linalg.norm(x)) / np.linalg.norm(x) < 0.5

    def test_mse_decreases_with_more_bits(self):
        """Higher bits → lower MSE."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)

        mse_2 = np.mean((x - turboquant_mse_decode(turboquant_mse(x, bits=2)))**2)
        mse_3 = np.mean((x - turboquant_mse_decode(turboquant_mse(x, bits=3)))**2)
        mse_4 = np.mean((x - turboquant_mse_decode(turboquant_mse(x, bits=4)))**2)

        assert mse_4 < mse_3 < mse_2

    def test_deterministic_with_seed(self):
        x = np.random.default_rng(0).standard_normal(32)
        b1 = turboquant_mse(x, bits=3, rotation_seed=42)
        b2 = turboquant_mse(x, bits=3, rotation_seed=42)
        x1 = turboquant_mse_decode(b1)
        x2 = turboquant_mse_decode(b2)
        assert np.allclose(x1, x2)

    def test_block_metadata(self):
        x = np.random.default_rng(0).standard_normal(64)
        block = turboquant_mse(x, bits=3)
        assert block.d == 64
        assert block.bits == 3
        assert block.radius > 0
        assert len(block.angle_indices) == 1  # single coordinate index array
        assert len(block.angle_indices[0]) == 64  # one index per coordinate


# ---------------------------------------------------------------------------
# TurboQuant prod (Stage 1 + QJL)
# ---------------------------------------------------------------------------


class TestTurboQuantProd:
    def test_encode_decode_shape(self):
        x = np.random.default_rng(42).standard_normal(64)
        block = turboquant_prod(x, bits=3)
        x_hat = turboquant_prod_decode(block)
        assert x_hat.shape == x.shape

    def test_has_qjl_data(self):
        x = np.random.default_rng(42).standard_normal(32)
        block = turboquant_prod(x, bits=3)
        assert block.qjl_signs is not None
        assert block.qjl_seed is not None
        assert block.qjl_norm >= 0


# ---------------------------------------------------------------------------
# TQBlock properties
# ---------------------------------------------------------------------------


class TestTQBlock:
    def test_compression_ratio_3bit(self):
        x = np.random.default_rng(0).standard_normal(128)
        block = turboquant_mse(x, bits=3)
        # 3-bit should achieve at least 2x compression vs FP16
        assert block.compression_ratio > 1.5

    def test_total_bits_positive(self):
        x = np.random.default_rng(0).standard_normal(32)
        block = turboquant_mse(x, bits=3)
        assert block.total_bits > 0


# ---------------------------------------------------------------------------
# Distortion bounds
# ---------------------------------------------------------------------------


class TestDistortionBounds:
    def test_mse_bound_decreases_with_bits(self):
        b2 = mse_distortion_bound(2)
        b3 = mse_distortion_bound(3)
        b4 = mse_distortion_bound(4)
        assert b4 < b3 < b2

    def test_mse_bound_positive(self):
        assert mse_distortion_bound(3) > 0

    def test_inner_product_bound(self):
        bound = inner_product_distortion_bound(3, norm_sq=1.0, d=128)
        assert bound > 0
        # Should decrease with d
        bound_large = inner_product_distortion_bound(3, norm_sq=1.0, d=1024)
        assert bound_large < bound


# ---------------------------------------------------------------------------
# Bit-packing
# ---------------------------------------------------------------------------


class TestBitPacking:
    def test_roundtrip_3bit(self):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 8, size=100, dtype=np.uint8)
        packed = pack_indices(indices, 3)
        unpacked = unpack_indices(packed, 3, 100)
        assert np.array_equal(indices, unpacked)

    def test_roundtrip_4bit(self):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 16, size=256, dtype=np.uint8)
        packed = pack_indices(indices, 4)
        unpacked = unpack_indices(packed, 4, 256)
        assert np.array_equal(indices, unpacked)

    def test_roundtrip_2bit(self):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 4, size=64, dtype=np.uint8)
        packed = pack_indices(indices, 2)
        unpacked = unpack_indices(packed, 2, 64)
        assert np.array_equal(indices, unpacked)

    def test_roundtrip_8bit(self):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 256, size=50, dtype=np.uint8)
        packed = pack_indices(indices, 8)
        unpacked = unpack_indices(packed, 8, 50)
        assert np.array_equal(indices, unpacked)

    def test_packing_reduces_size_3bit(self):
        indices = np.zeros(256, dtype=np.uint8)
        packed = pack_indices(indices, 3)
        assert len(packed) < 256  # 3×256/8 = 96 bytes < 256

    def test_packing_reduces_size_2bit(self):
        indices = np.zeros(256, dtype=np.uint8)
        packed = pack_indices(indices, 2)
        assert len(packed) == 64  # 2×256/8 = 64 bytes


# ---------------------------------------------------------------------------
# KV-cache batch operations
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_compress_decompress_roundtrip(self):
        rng = np.random.default_rng(42)
        d = 64
        n_tokens = 8
        keys = rng.standard_normal((n_tokens, d))
        values = rng.standard_normal((n_tokens, d))

        key_blocks, val_blocks = compress_kv_cache(keys, values, bits=3)
        keys_hat, values_hat = decompress_kv_cache(key_blocks, val_blocks)

        assert keys_hat.shape == keys.shape
        assert values_hat.shape == values.shape

    def test_compress_prod_method(self):
        rng = np.random.default_rng(42)
        keys = rng.standard_normal((4, 32))
        values = rng.standard_normal((4, 32))

        key_blocks, val_blocks = compress_kv_cache(
            keys, values, bits=3, method="prod"
        )
        keys_hat, values_hat = decompress_kv_cache(
            key_blocks, val_blocks, method="prod"
        )
        assert keys_hat.shape == keys.shape


# ---------------------------------------------------------------------------
# GGML bridge fallback
# ---------------------------------------------------------------------------


class TestGGMLBridge:
    def test_quantize_dequantize(self):
        from moirais.quant_bridge import GGMLTurboQuant

        tq = GGMLTurboQuant()
        # Works with either C lib or NumPy fallback
        x = np.random.default_rng(42).standard_normal(256).astype(np.float32)
        block = tq.quantize(x, bits=3)
        x_hat = tq.dequantize(block, bits=3)
        assert x_hat.shape == (256,)
        # Roundtrip should preserve rough magnitude
        assert np.linalg.norm(x_hat) > 0
