"""Tests for moirais.fn.amfit — A-M fit statistic."""
import numpy as np
from moirais.fn.amfit import amfit


def test_amfit_smoke():
    zhat = np.array([1.0, 3.0, 5.0])
    alpha = np.array([0.5, 0.0])
    beta = np.array([1.0, -1.0])
    Z = alpha[:, None] + beta[:, None] * zhat[None, :]
    r = amfit(Z, zhat, alpha, beta)
    assert r.name == "am_fit_statistic"
    assert abs(r.value - 1.0) < 1e-6


def test_cheatsheet():
    from moirais.fn.amfit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
