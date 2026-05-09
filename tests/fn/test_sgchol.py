"""Tests for Cholesky GRF simulation."""
import numpy as np
from moirais.fn.sgchol import sgchol


def test_sgchol_smoke():
    rng = np.random.default_rng(22)
    coords = rng.uniform(0, 5, (20, 2))
    r = sgchol(coords, n_sims=3)
    assert r.name == "cholesky_grf_sim"
    assert r.extra["simulations"].shape == (3, 20)


def test_sgchol_gaussian():
    coords = np.column_stack([np.linspace(0, 5, 10), np.zeros(10)])
    r = sgchol(coords, cov_model="gaussian", n_sims=1)
    assert r.extra["simulations"].shape == (1, 10)
