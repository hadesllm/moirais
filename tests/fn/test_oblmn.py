"""Tests for moirais.fn.oblmn -- Oblimin rotation."""

import numpy as np
from moirais.fn.oblmn import oblimin, oblmn
from moirais.fn._containers import DescriptiveResult


class TestOblimin:
    def test_alias(self):
        assert oblmn is oblimin

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((6, 2))
        res = oblimin(L)
        assert isinstance(res, DescriptiveResult)

    def test_has_phi(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((6, 2))
        res = oblimin(L)
        assert "phi" in res.extra

    def test_same_shape(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((5, 3))
        res = oblimin(L)
        assert res.value.shape == (5, 3)
