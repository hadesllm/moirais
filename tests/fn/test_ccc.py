"""Tests for concordance_corr."""
import numpy as np, pytest
from moirais.fn.ccc import concordance_corr

class TestCCC:
    def test_perfect(self):
        x = np.arange(10, dtype=float)
        r = concordance_corr(x, x)
        assert r.estimate == pytest.approx(1.0, abs=0.001)

    def test_shifted(self):
        x = np.arange(10, dtype=float)
        y = x + 5
        r = concordance_corr(x, y)
        assert r.estimate < 1.0
