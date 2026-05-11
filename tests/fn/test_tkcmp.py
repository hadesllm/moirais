"""Test tucker_decompose (tkcmp)."""
import numpy as np
import pytest
from morie.fn.tkcmp import tucker_decompose, tkcmp
from morie.fn._containers import DescriptiveResult


class TestTkcmp:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 4, 3))
        result = tucker_decompose(X, ranks=(2, 2, 2))
        assert isinstance(result, DescriptiveResult)
        assert result.name == "tucker_decompose"
        assert result.value >= 0

    def test_core_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 5, 4))
        r = tucker_decompose(X, ranks=(3, 2, 2))
        assert r.extra["core"].shape == (3, 2, 2)
        A, B, C = r.extra["factors"]
        assert A.shape == (6, 3)
        assert B.shape == (5, 2)
        assert C.shape == (4, 2)

    def test_full_rank_exact(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 3, 3))
        r = tucker_decompose(X, ranks=(3, 3, 3))
        assert r.value < 1e-6

    def test_not_3d_raises(self):
        with pytest.raises(ValueError):
            tucker_decompose(np.ones((3, 4)))

    def test_alias(self):
        assert tkcmp is tucker_decompose
