"""Tests for morie.fn.prmxr -- Promax rotation."""

import numpy as np
from morie.fn.prmxr import promax, prmxr
from morie.fn._containers import DescriptiveResult


class TestPromax:
    def test_alias(self):
        assert prmxr is promax

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((6, 2))
        res = promax(L)
        assert isinstance(res, DescriptiveResult)

    def test_has_phi(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((8, 3))
        res = promax(L)
        assert "phi" in res.extra

    def test_same_shape(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((5, 2))
        res = promax(L)
        assert res.value.shape == (5, 2)
