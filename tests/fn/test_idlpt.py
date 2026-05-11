"""Tests for morie.fn.idlpt — Ideal point estimation."""
import numpy as np
import pytest

from morie.fn.idlpt import idlpt


def test_idlpt_smoke():
    X_r = np.array([[1, 2], [3, 4]])
    X_s = np.array([[0, 0]])
    r = idlpt(X_r, X_s)
    assert r.extra["ideal_points"].shape == (2, 2)


def test_cheatsheet():
    from morie.fn.idlpt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
