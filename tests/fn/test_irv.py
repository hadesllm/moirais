"""Tests for morie.fn.irv — Instant runoff voting."""
import numpy as np
from morie.fn.irv import irv


def test_irv_majority():
    rankings = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3]])
    r = irv(rankings)
    assert r.value == 0


def test_irv_elimination():
    rankings = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1], [3, 1, 2], [2, 3, 1]])
    r = irv(rankings)
    assert isinstance(r.value, int)
    assert len(r.extra["elimination_order"]) >= 1


def test_irv_two_candidates():
    rankings = np.array([[1, 2], [2, 1], [1, 2]])
    r = irv(rankings)
    assert r.value == 0
