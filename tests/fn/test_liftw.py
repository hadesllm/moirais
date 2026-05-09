"""Tests for liftw.py - Lifting scheme DWT."""
import numpy as np
from moirais.fn.liftw import lifting_dwt, liftw


def test_liftw_returns_descriptive_result():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = lifting_dwt(x, wavelet="haar")
    assert result.name == "lifting_dwt"
    assert "coeffs" in result.extra
    assert result.extra["method"] == "lifting"


def test_liftw_haar_reconstruction():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = lifting_dwt(x, wavelet="haar")
    a = result.extra["coeffs"][0]
    d = result.extra["coeffs"][1]
    rec = np.zeros(len(a) * 2)
    rec[0::2] = a - d / 2.0
    rec[1::2] = a - d / 2.0 + d
    np.testing.assert_allclose(rec, x, atol=1e-10)


def test_liftw_alias():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = liftw(x, wavelet="haar")
    assert result.name == "lifting_dwt"
