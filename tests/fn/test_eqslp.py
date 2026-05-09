"""Tests for moirais.fn.eqslp — slope inequality index."""

import pytest
from moirais.fn.eqslp import slope_inequality
from moirais.fn._containers import ESRes


class TestSlopeInequality:
    def test_positive_gradient(self):
        r = slope_inequality([10, 8, 6, 4, 2], [0.1, 0.3, 0.5, 0.7, 0.9])
        assert isinstance(r, ESRes)
        assert r.estimate < 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            slope_inequality([1, 2], [0.3, 0.7])
