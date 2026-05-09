"""Tests for spatial data decomposition."""
import numpy as np
from moirais.fn.sgdcp import sgdcp


def test_sgdcp_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = 2.0 * coords[:, 0] + 3.0 * coords[:, 1] + rng.normal(0, 0.5, 50)
    r = sgdcp(Z, coords, trend_order=1)
    assert r.name == "data_decomposition"
    assert "trend" in r.extra
    assert "residuals" in r.extra


def test_sgdcp_residuals_small():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = 2.0 * coords[:, 0] + 3.0 * coords[:, 1]
    r = sgdcp(Z, coords, trend_order=1)
    assert np.max(np.abs(r.extra["residuals"])) < 1e-10
