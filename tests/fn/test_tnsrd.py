"""Test tensor_decompose (tnsrd)."""
import numpy as np
from morie.fn.tnsrd import tensor_decompose, tnsrd
from morie.fn._containers import DescriptiveResult


class TestTnsrd:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 6, 7))
        result = tensor_decompose(X, rank=2, max_iter=30, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "tensor_decompose"
        assert result.value >= 0.0

    def test_factors_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 5, 6))
        r = tensor_decompose(X, rank=3, max_iter=20, seed=0)
        A, B, C = r.extra["factors"]
        assert A.shape == (4, 3)
        assert B.shape == (5, 3)
        assert C.shape == (6, 3)

    def test_alias(self):
        assert tnsrd is tensor_decompose
