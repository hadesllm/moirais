"""Tests for mexht.py - Mexican hat wavelet."""
import numpy as np
from moirais.fn.mexht import mexican_hat, mexht


def test_mexht_returns_descriptive_result():
    result = mexican_hat(N=256)
    assert result.name == "mexican_hat"
    assert "wavelet" in result.extra
    assert "time" in result.extra


def test_mexht_zero_mean():
    result = mexican_hat(N=1024)
    psi = result.extra["wavelet"]
    np.testing.assert_allclose(np.mean(psi), 0.0, atol=0.01)


def test_mexht_alias():
    result = mexht(N=64)
    assert result.name == "mexican_hat"
