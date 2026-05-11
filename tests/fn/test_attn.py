"""Tests for morie.fn.attn — scaled dot-product attention."""
import numpy as np
import pytest
from morie.fn.attn import attention


class TestAttention:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((4, 8))
        K = rng.standard_normal((4, 8))
        V = rng.standard_normal((4, 6))
        res = attention(Q, K, V)
        output = res.extra["output"]
        assert output.shape == (4, 6)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((4, 8))
        K = rng.standard_normal((4, 8))
        V = rng.standard_normal((4, 8))
        res = attention(Q, K, V)
        weights = res.extra["attention_weights"]
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_single_query(self):
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((1, 8))
        K = rng.standard_normal((5, 8))
        V = rng.standard_normal((5, 4))
        res = attention(Q, K, V)
        assert res.extra["output"].shape == (1, 4)
