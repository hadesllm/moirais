"""Tests for morie.fn.svd_ — truncated SVD."""
import numpy as np
import pytest
from morie.fn.svd_ import truncated_svd


class TestTruncatedSVD:
    def test_basic(self):
        X = np.random.default_rng(42).standard_normal((20, 5))
        res = truncated_svd(X, k=2)
        assert len(res.extra["S"]) == 2
        assert res.value <= 1.0 + 1e-10

    def test_full_rank(self):
        X = np.random.default_rng(42).standard_normal((10, 3))
        res = truncated_svd(X, k=3)
        assert res.value == pytest.approx(1.0, abs=1e-10)
