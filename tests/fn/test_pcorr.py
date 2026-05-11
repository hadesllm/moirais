"""Tests for partial_correlation."""
import numpy as np, pytest
from morie.fn.pcorr import partial_correlation

class TestPcorr:
    def test_controlled(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 50)
        x = z + rng.normal(0, 0.5, 50)
        y = z + rng.normal(0, 0.5, 50)
        r = partial_correlation(x, y, z)
        assert abs(r.estimate) < abs(np.corrcoef(x, y)[0, 1])

    def test_returns_pvalue(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)
        z = rng.normal(0, 1, 30)
        r = partial_correlation(x, y, z)
        assert "p_value" in r.extra
