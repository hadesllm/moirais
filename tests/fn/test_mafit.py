"""Test mafit."""
import numpy as np
import pytest
from morie.fn.mafit import mafit


def test_mafit_basic():
    """MA(1) fitting."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(100)
    # MA(1) with theta = -0.5
    y = np.zeros(100)
    y[0] = eps[0]
    for t in range(1, 100):
        y[t] = eps[t] - 0.5 * eps[t-1]

    r = mafit(y, q=1)
    assert isinstance(r.ma_coeff, np.ndarray)
    assert r.ma_coeff.shape == (1,)
    assert r.sigma2 > 0


def test_mafit_higher_order():
    """MA(2) model."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(150)
    y = np.zeros(150)
    y[0] = eps[0]
    y[1] = eps[1]
    for t in range(2, 150):
        y[t] = eps[t] - 0.3 * eps[t-1] + 0.2 * eps[t-2]

    r = mafit(y, q=2)
    assert r.ma_coeff.shape == (2,)
    assert r.acf.shape == (3,)  # q+1


def test_mafit_acf_properties():
    """ACF properties."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    r = mafit(y, q=2)

    assert r.acf[0] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.abs(r.acf) <= 1.0)


def test_mafit_validation():
    """Input validation."""
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        mafit(y, q=3)  # q >= n
