"""Tests for haarf.py - Haar wavelet transform."""
import numpy as np
from moirais.fn.haarf import haar_transform, haarf


def test_haarf_returns_descriptive_result():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = haar_transform(x)
    assert result.name == "haar_transform"
    assert "approximation" in result.extra
    assert "detail" in result.extra


def test_haarf_energy_preservation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = haar_transform(x)
    a = result.extra["approximation"]
    d = result.extra["detail"]
    np.testing.assert_allclose(np.sum(x**2), np.sum(a**2) + np.sum(d**2), atol=1e-10)


def test_haarf_alias():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = haarf(x)
    assert result.name == "haar_transform"
