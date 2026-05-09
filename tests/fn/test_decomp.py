"""Tests for moirais.fn.decomp — Seasonal decomposition."""
import numpy as np
import pytest

from moirais.fn.decomp import seasonal_decompose, decomp


def test_seasonal_signal():
    """Known seasonal pattern should be recovered."""
    rng = np.random.default_rng(42)
    n = 120
    t = np.arange(n, dtype=float)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.1 * t
    x = trend + seasonal + rng.standard_normal(n) * 0.5
    result = seasonal_decompose(x, period=12)
    assert result.extra["period"] == 12
    assert result.extra["model"] == "additive"
    # Seasonal component should have non-trivial amplitude
    s = np.array(result.extra["seasonal"])
    assert np.std(s) > 1.0


def test_too_short_raises():
    with pytest.raises(ValueError):
        seasonal_decompose(np.ones(10), period=12)


def test_decomp_alias():
    assert decomp is seasonal_decompose
