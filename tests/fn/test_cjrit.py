"""Tests for moirais.fn.cjrit — Bayesian IRT for roll call."""
import numpy as np
import pytest

from moirais.fn.cjrit import cjrit


def test_cjrit_smoke():
    votes = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]], dtype=float)
    r = cjrit(votes, n_dims=1, n_samples=50, burn_in=10)
    assert "ideal_point_mean" in r.extra


def test_cheatsheet():
    from moirais.fn.cjrit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
