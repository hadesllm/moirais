"""Tests for morie.fn.borda — Borda count."""
import numpy as np
from morie.fn.borda import borda


def test_borda_simple():
    rankings = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3]])
    r = borda(rankings)
    assert r.value == 0


def test_borda_tie():
    rankings = np.array([[1, 2], [2, 1]])
    r = borda(rankings)
    assert r.extra["scores"][0] == r.extra["scores"][1]


def test_borda_unanimous():
    rankings = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    r = borda(rankings)
    assert r.value == 0
    assert r.extra["scores"][0] > r.extra["scores"][1]
