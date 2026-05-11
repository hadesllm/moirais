"""Tests for boolm (boolean minimization)."""
import numpy as np
from morie.fn.boolm import boolean_minimize


def test_boolean_minimize_basic():
    tt = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    r = boolean_minimize(tt)
    assert "prime_implicants" in r.extra


def test_cheatsheet():
    from morie.fn.boolm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
