"""Tests for dfbetas."""
import numpy as np, pytest
from morie.fn.dfbts import dfbetas

class TestDFBETAS:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 2))
        y = X[:, 0] + rng.normal(0, 0.1, 40)
        r = dfbetas(X, y)
        assert r.name == "dfbetas"
        assert len(r.extra["max_abs_dfbetas_per_coef"]) == 3

    def test_outlier_detected(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (30, 1))
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        y[0] = 100
        r = dfbetas(X, y)
        assert r.extra["n_influential"] >= 1
