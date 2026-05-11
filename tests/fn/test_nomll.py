"""Tests for morie.fn.nomll — NOMINATE log-likelihood."""
import numpy as np
import pytest

from morie.fn.nomll import nomll


def test_nomll_smoke():
    votes_sm = np.array([[1, 0], [0, 1]])
    x = np.array([[0.5], [-0.5]])
    zy = np.array([[0.3], [0.2]])
    zn = np.array([[-0.3], [-0.2]])
    r = nomll(votes_sm, x, zy, zn)
    assert r.extra["loglik"] < 0


def test_cheatsheet():
    from morie.fn.nomll import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
