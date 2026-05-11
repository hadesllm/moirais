"""Tests for morie.fn.bblck — Blackbox scaling."""
import numpy as np
import pytest

from morie.fn.bblck import bblck


def test_bblck_smoke():
    Z = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 5.0], [1.5, 3.5, 4.5], [0.5, 2.5, 4.0]])
    r = bblck(Z)
    assert r.name == "blackbox_scaling"
    assert "ideal_points" in r.extra


def test_cheatsheet():
    from morie.fn.bblck import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
