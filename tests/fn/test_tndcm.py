"""Test tensor_decompose (tndcm)."""
import numpy as np
import pytest
from moirais.fn.tndcm import tensor_decompose, tndcm
from moirais.fn._containers import DescriptiveResult


class TestTndcm:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 5, 4))
        result = tensor_decompose(X, rank=2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "tensor_decompose"
        assert result.value >= 0

    def test_rank1(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        c = np.array([6.0, 7.0])
        X = np.einsum("i,j,k->ijk", a, b, c)
        r = tensor_decompose(X, rank=1, max_iter=200)
        assert r.value < 0.1

    def test_factors_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((4, 5, 3))
        r = tensor_decompose(X, rank=2)
        A, B, C = r.extra["factors"]
        assert A.shape == (4, 2)
        assert B.shape == (5, 2)
        assert C.shape == (3, 2)

    def test_not_3d_raises(self):
        with pytest.raises(ValueError):
            tensor_decompose(np.ones((3, 4)))

    def test_alias(self):
        assert tndcm is tensor_decompose
