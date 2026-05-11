"""Tests for morie.engine_bridge — C kernel bridge with Accelerate.framework."""

import numpy as np
import pytest

from morie.engine_bridge import (
    argmax,
    is_available,
    matvec,
    rmsnorm,
    rope,
    silu_inplace,
    softmax,
)


class TestAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)


class TestRMSNorm:
    def test_output_rms_is_one(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256).astype(np.float32)
        w = np.ones(256, dtype=np.float32)
        out = rmsnorm(x, w)
        rms = np.sqrt(np.mean(out ** 2))
        assert abs(rms - 1.0) < 0.01

    def test_weight_scaling(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        w = np.array([2.0, 2.0], dtype=np.float32)
        out = rmsnorm(x, w)
        # With unit weight, rms(out) = 1; with w=2, rms(out) = 2
        rms = np.sqrt(np.mean(out ** 2))
        assert abs(rms - 2.0) < 0.01

    def test_preserves_shape(self):
        x = np.ones(128, dtype=np.float32)
        w = np.ones(128, dtype=np.float32)
        assert rmsnorm(x, w).shape == (128,)


class TestRoPE:
    def test_preserves_norm(self):
        rng = np.random.default_rng(42)
        q = rng.standard_normal(64).astype(np.float32)
        k = rng.standard_normal(64).astype(np.float32)
        q_norm = np.linalg.norm(q)
        k_norm = np.linalg.norm(k)
        q_r, k_r = rope(q, k, position=10, head_dim=64)
        assert abs(np.linalg.norm(q_r) - q_norm) < 0.01
        assert abs(np.linalg.norm(k_r) - k_norm) < 0.01

    def test_different_positions_differ(self):
        q = np.ones(64, dtype=np.float32)
        k = np.ones(64, dtype=np.float32)
        q0, _ = rope(q.copy(), k.copy(), position=0, head_dim=64)
        q1, _ = rope(q.copy(), k.copy(), position=1, head_dim=64)
        assert not np.allclose(q0, q1)

    def test_position_zero_identity_for_q(self):
        q = np.ones(64, dtype=np.float32)
        k = np.ones(64, dtype=np.float32)
        q_r, _ = rope(q, k, position=0, head_dim=64)
        # At position 0, all angles are 0, so cos=1, sin=0
        np.testing.assert_allclose(q_r, q, atol=1e-5)


class TestMatvec:
    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((64, 128)).astype(np.float32)
        x = rng.standard_normal(128).astype(np.float32)
        out = matvec(A, x)
        expected = A @ x
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_identity_matrix(self):
        I = np.eye(32, dtype=np.float32)
        x = np.arange(32, dtype=np.float32)
        np.testing.assert_allclose(matvec(I, x), x, atol=1e-6)


class TestSiLU:
    def test_zero_maps_to_zero(self):
        x = np.array([0.0], dtype=np.float32)
        assert abs(silu_inplace(x)[0]) < 1e-6

    def test_large_positive(self):
        x = np.array([10.0], dtype=np.float32)
        out = silu_inplace(x)
        # SiLU(10) ≈ 10 * sigmoid(10) ≈ 10 * 1 ≈ 10
        assert abs(out[0] - 10.0) < 0.01

    def test_negative(self):
        x = np.array([-10.0], dtype=np.float32)
        out = silu_inplace(x)
        # SiLU(-10) ≈ -10 * sigmoid(-10) ≈ -10 * 0 ≈ 0
        assert abs(out[0]) < 0.01


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(softmax(x).sum() - 1.0) < 1e-5

    def test_monotone(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = softmax(x)
        assert p[2] > p[1] > p[0]

    def test_large_values_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        p = softmax(x)
        assert abs(p.sum() - 1.0) < 1e-5
        assert not np.any(np.isnan(p))


class TestArgmax:
    def test_basic(self):
        assert argmax(np.array([0.1, 0.9, 0.5], dtype=np.float32)) == 1

    def test_first_element(self):
        assert argmax(np.array([9.0, 1.0, 2.0], dtype=np.float32)) == 0

    def test_last_element(self):
        assert argmax(np.array([1.0, 2.0, 9.0], dtype=np.float32)) == 2
