"""Tests for tukey_biweight."""
import numpy as np, pytest
from moirais.fn.tukey import tukey_biweight

class TestTukey:
    def test_normal(self):
        rng = np.random.default_rng(7)
        x = rng.normal(10, 2, 300)
        r = tukey_biweight(x)
        assert r.estimate == pytest.approx(10.0, abs=0.5)

    def test_outlier_robust(self):
        x = np.concatenate([np.ones(50) * 5, [500]])
        r = tukey_biweight(x)
        assert r.estimate == pytest.approx(5.0, abs=1.0)
