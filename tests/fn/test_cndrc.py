"""Tests for morie.fn.cndrc — Condorcet winner."""
import numpy as np
from morie.fn.cndrc import cndrc


def test_cndrc_winner():
    M = np.array([[0, 3, 3], [1, 0, 3], [1, 1, 0]])
    r = cndrc(M)
    assert r.name == "condorcet_winner"
    assert r.value == 0
    assert r.extra["has_winner"] is True


def test_cndrc_cycle():
    M = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    r = cndrc(M)
    assert r.value == -1
