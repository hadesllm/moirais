"""Tests for empirical semivariogram."""
import numpy as np
from morie.fn.sgemp import sgemp


def test_sgemp_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = rng.normal(0, 1, 50)
    r = sgemp(Z, coords, n_lags=10)
    assert r.name == "empirical_semivariogram"
    assert "gamma_values" in r.extra
    assert len(r.extra["gamma_values"]) == 10


def test_sgemp_increasing():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (80, 2))
    Z = coords[:, 0] + rng.normal(0, 0.1, 80)
    r = sgemp(Z, coords, n_lags=5)
    gamma = r.extra["gamma_values"]
    assert gamma[-1] >= gamma[0]
