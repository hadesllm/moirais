"""Tests for morie.fn.pairm — Pairwise matrix."""
import numpy as np
from morie.fn.pairm import pairm


def test_pairm_basic():
    rankings = np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3]])
    r = pairm(rankings)
    M = r.value["matrix"]
    assert M[0, 1] == 2
    assert M[1, 0] == 1


def test_pairm_unanimous():
    rankings = np.array([[1, 2], [1, 2], [1, 2]])
    r = pairm(rankings)
    M = r.value["matrix"]
    assert M[0, 1] == 3
    assert M[1, 0] == 0


def test_pairm_symmetric_n():
    rankings = np.array([[1, 2, 3], [3, 2, 1]])
    r = pairm(rankings)
    M = r.value["matrix"]
    assert M[0, 2] + M[2, 0] == 2
