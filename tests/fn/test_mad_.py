"""Tests for median_abs_deviation."""
import numpy as np, pytest
from moirais.fn.mad_ import median_abs_deviation

class TestMAD:
    def test_normal(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        r = median_abs_deviation(x)
        assert r.estimate == pytest.approx(1.0, abs=0.2)

    def test_constant(self):
        x = np.ones(10)
        r = median_abs_deviation(x)
        assert r.estimate == pytest.approx(0.0, abs=1e-8)
