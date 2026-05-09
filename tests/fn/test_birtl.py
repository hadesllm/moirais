"""Tests for moirais.fn.birtl — Bayesian IRT log-likelihood."""
import numpy as np
import pytest

from moirais.fn.birtl import birtl


def test_birtl_smoke():
    votes = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]], dtype=float)
    x = np.array([[0.5], [-0.5], [0.0], [0.3], [-0.3]])
    alpha = np.array([0.0, 0.1, -0.1, 0.0])
    beta = np.array([[1.0], [0.5], [0.8], [1.2]])
    r = birtl(votes, x, alpha, beta)
    assert r.extra["loglik"] < 0


def test_cheatsheet():
    from moirais.fn.birtl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
