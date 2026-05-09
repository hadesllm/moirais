"""Tests for moirais.fn.amres — A-M residuals."""
import numpy as np
from moirais.fn.amres import amres


def test_amres_perfect():
    zhat = np.array([1.0, 3.0, 5.0])
    alpha = np.array([0.5, 0.0])
    beta = np.array([1.0, 1.0])
    Z = alpha[:, None] + beta[:, None] * zhat[None, :]
    r = amres(Z, zhat, alpha, beta)
    assert r.name == "am_residuals"
    assert r.value < 1e-10


def test_cheatsheet():
    from moirais.fn.amres import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
