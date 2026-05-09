"""Tests for moirais.fn.appvl — Approval voting."""
import numpy as np
from moirais.fn.appvl import appvl


def test_appvl_basic():
    A = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
    r = appvl(A)
    assert r.value == 0
    assert r.extra["counts"] == [3, 2, 2]


def test_appvl_single():
    A = np.array([[0, 1], [0, 1], [1, 0]])
    r = appvl(A)
    assert r.value == 1


def test_appvl_rate():
    A = np.array([[1, 0], [1, 0]])
    r = appvl(A)
    assert r.extra["winner_approval_rate"] == 1.0
