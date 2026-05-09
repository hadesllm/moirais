"""Tests for moirais.fn.renrg -- renormalization group beta function."""

import numpy as np
import pytest

from moirais.fn.renrg import renrg


def test_returns_dict():
    r = renrg(coupling=1.0)
    assert isinstance(r, dict)
    for k in ("beta", "b0", "asymptotic_freedom", "alpha_s"):
        assert k in r


def test_asymptotic_freedom_qcd():
    r = renrg(coupling=1.0, n_flavors=6, n_colors=3)
    assert r["asymptotic_freedom"] is True


def test_beta_negative_qcd():
    r = renrg(coupling=1.0, n_flavors=6, n_colors=3)
    assert r["beta"] < 0


def test_alpha_s():
    g = 2.0
    r = renrg(coupling=g)
    assert r["alpha_s"] == pytest.approx(g ** 2 / (4 * np.pi), rel=1e-10)


def test_two_loop():
    r = renrg(coupling=1.0, loop_order=2)
    assert "b1" in r


def test_negative_coupling_raises():
    with pytest.raises(ValueError):
        renrg(coupling=-1.0)
