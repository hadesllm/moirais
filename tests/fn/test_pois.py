"""Tests for poisson_regression."""
import numpy as np, pytest
from moirais.fn.pois import poisson_regression

class TestPoisson:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 1))
        lam = np.exp(0.5 + 0.3 * X[:, 0])
        y = rng.poisson(lam)
        r = poisson_regression(y, X)
        assert r.name == "poisson"
        assert r.extra["coefficients"]["x0"] == pytest.approx(0.3, abs=0.3)

    def test_deviance_positive(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 1))
        y = rng.poisson(3, 50)
        r = poisson_regression(y, X)
        assert r.value > 0
