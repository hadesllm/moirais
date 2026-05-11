"""Test arfit."""
import numpy as np
import pytest
from morie.fn.arfit import arfit


def test_arfit_basic():
    """AR(1) on simulated data."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(200)
    # AR(1) with phi=0.7
    y = np.zeros(200)
    y[0] = eps[0]
    for t in range(1, 200):
        y[t] = 0.7 * y[t-1] + eps[t]

    r = arfit(y, p=1)
    assert isinstance(r.ar_coeff, np.ndarray)
    assert r.ar_coeff.shape == (1,)
    assert 0.5 < r.ar_coeff[0] < 0.9  # Should recover ~0.7
    assert r.sigma2 > 0
    assert np.isfinite(r.aic) and np.isfinite(r.bic)


def test_arfit_demean():
    """Test demean parameter."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(100)
    y = np.cumsum(eps) + 5.0  # Non-zero mean

    r_demean = arfit(y, p=1, demean=True)
    r_no_demean = arfit(y, p=1, demean=False)

    assert r_demean.sigma2 > 0
    assert r_no_demean.sigma2 > 0


def test_arfit_higher_order():
    """AR(2) model."""
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(300)
    y = np.zeros(300)
    y[0] = eps[0]
    y[1] = eps[1]
    for t in range(2, 300):
        y[t] = 0.5 * y[t-1] - 0.2 * y[t-2] + eps[t]

    r = arfit(y, p=2)
    assert r.ar_coeff.shape == (2,)
    assert r.acf.shape == (3,)  # p+1
    assert r.pacf.shape == (3,)  # p+1


def test_arfit_validation():
    """Input validation."""
    y = np.array([1, 2, 3])

    # p >= n should fail
    with pytest.raises(ValueError):
        arfit(y, p=3)

    # 2-D should fail
    with pytest.raises(ValueError):
        arfit(np.array([[1, 2], [3, 4]]), p=1)


def test_arfit_acf_pacf():
    """ACF and PACF properties."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(150)
    r = arfit(y, p=3)

    assert r.acf[0] == pytest.approx(1.0, abs=1e-10)  # ACF(0) = 1
    assert r.pacf[0] == pytest.approx(1.0, abs=1e-10)  # PACF(0) = 1
    assert np.all(np.isfinite(r.acf))
    assert np.all(np.isfinite(r.pacf))


def test_arfit_stats():
    """Statistics consistency."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)
    r = arfit(y, p=2)

    assert r.n == 100
    assert r.p == 2
    assert r.bic > r.aic  # bic=p*log(n) > aic=2p for n>e²≈7.4
