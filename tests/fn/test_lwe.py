"""Test lwe_sample."""
import numpy as np
import pytest
from moirais.fn.lwe import lwe_sample
from moirais.fn._containers import CryptoResult


class TestLweSample:
    def test_basic(self):
        result = lwe_sample(n=4, m=8, q=97)
        assert result.success is True

    def test_output_type(self):
        result = lwe_sample(n=4, m=8, q=97)
        assert isinstance(result, CryptoResult)

    def test_extra_keys(self):
        result = lwe_sample(n=4, m=8, q=97)
        for key in ("A", "b", "s", "e"):
            assert key in result.extra

    def test_shapes(self):
        n, m = 5, 10
        result = lwe_sample(n=n, m=m, q=97)
        A = np.asarray(result.extra["A"])
        b = np.asarray(result.extra["b"])
        s = np.asarray(result.extra["s"])
        assert A.shape == (m, n)
        assert b.shape[0] == m
        assert s.shape[0] == n
