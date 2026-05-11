"""Tests for morie.fn.copld — Copeland method."""
import numpy as np
from morie.fn.copld import copld


def test_copld_clear_winner():
    M = np.array([[0, 3, 3], [1, 0, 3], [1, 1, 0]])
    r = copld(M)
    assert r.value == 0
    assert r.extra["scores"][0] > r.extra["scores"][1]


def test_copld_cycle():
    M = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    r = copld(M)
    assert all(s == 0 for s in r.extra["scores"])


def test_copld_two():
    M = np.array([[0, 5], [3, 0]])
    r = copld(M)
    assert r.value == 0
