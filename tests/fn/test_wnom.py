"""Tests for morie.fn.wnom — W-NOMINATE scaling."""
import numpy as np
import pytest

from morie.fn.wnom import wnom


def test_wnom_smoke():
    votes = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]], dtype=float)
    x = np.array([[0.5], [-0.5], [0.0], [0.3], [-0.3]])
    zy = np.array([[0.3], [0.2], [0.1], [0.0]])
    zn = np.array([[-0.3], [-0.2], [-0.1], [0.0]])
    r = wnom(votes, x, zy, zn)
    assert "GMP" in r.extra


def test_cheatsheet():
    from morie.fn.wnom import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
