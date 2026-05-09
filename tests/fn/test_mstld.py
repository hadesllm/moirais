"""Test mstld."""
import numpy as np
import pytest
from moirais.fn.mstld import mstld


def test_mstld_basic():
    """MSTL decomposition with default period."""
    rng = np.random.default_rng(42)
    y = np.sin(np.arange(144) * 2 * np.pi / 12) + rng.standard_normal(144) * 0.1
    r = mstld(y)
    assert r.trend.shape == y.shape
    assert len(r.seasonals) > 0
    assert r.remainder.shape == y.shape


def test_mstld_multiple_periods():
    """MSTL with multiple seasonal periods."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(365)
    r = mstld(y, periods=[7, 365])
    assert len(r.seasonals) == 2


def test_mstld_reconstruction():
    """Verify decomposition sum."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    r = mstld(y, periods=[12])
    # Trend + seasonals + remainder ≈ y
    reconstructed = r.trend + np.sum([s for s in r.seasonals], axis=0) + r.remainder
    assert np.allclose(reconstructed, y, atol=1e-10)
