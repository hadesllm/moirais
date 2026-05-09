"""Tests for moirais.fn.bunfl — Bayesian unfolding."""
import numpy as np
import pytest

from moirais.fn.bunfl import bunfl


def test_bunfl_smoke():
    D_u = np.random.default_rng(42).random((4, 3)) + 0.5
    r = bunfl(D_u, n_dims=1, n_samples=50, burn_in=10)
    assert "respondent_mean" in r.extra


def test_cheatsheet():
    from moirais.fn.bunfl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
