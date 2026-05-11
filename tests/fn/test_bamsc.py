"""Tests for morie.fn.bamsc — Bayesian Aldrich-McKelvey scaling."""
import numpy as np
import pytest

from morie.fn.bamsc import bamsc


def test_bamsc_smoke():
    Z = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 5.0], [1.5, 3.5, 4.5], [0.5, 2.5, 4.0]])
    r = bamsc(Z, n_samples=50, burn_in=10)
    assert "zeta_mean" in r.extra


def test_cheatsheet():
    from morie.fn.bamsc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
