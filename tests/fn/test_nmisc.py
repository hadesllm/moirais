"""Tests for moirais.fn.nmisc -- Normalized mutual information."""

import numpy as np
from moirais.fn.nmisc import nmi, nmisc
from moirais.fn._containers import DescriptiveResult


class TestNmi:
    def test_alias(self):
        assert nmisc is nmi

    def test_returns_result(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        yp = np.array([0, 0, 1, 1, 2, 2])
        res = nmi(y, yp)
        assert isinstance(res, DescriptiveResult)

    def test_perfect_agreement(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        res = nmi(y, y)
        assert abs(res.value - 1.0) < 1e-6

    def test_bounded(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 3, 30)
        yp = rng.integers(0, 3, 30)
        res = nmi(y, yp)
        assert 0 <= res.value <= 1.0
