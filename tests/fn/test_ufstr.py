"""Tests for moirais.fn.ufstr — Unfolding stress."""
import numpy as np
import pytest

from moirais.fn.ufstr import ufstr


def test_ufstr_smoke():
    X_r = np.random.default_rng(42).standard_normal((4, 2))
    X_s = np.random.default_rng(43).standard_normal((3, 2))
    D_u = np.random.default_rng(44).random((4, 3)) + 0.5
    r = ufstr(X_r, X_s, D_u)
    assert r.value >= 0


def test_cheatsheet():
    from moirais.fn.ufstr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
