"""Tests for moirais.fn.btlmd — Bradley-Terry-Luce model."""
import numpy as np
from moirais.fn.btlmd import btlmd


def test_btlmd_basic():
    W = np.array([[0, 5, 3], [1, 0, 4], [3, 2, 0]], dtype=float)
    r = btlmd(W)
    assert len(r.value["strengths"]) == 3
    assert abs(sum(r.value["strengths"]) - 1.0) < 1e-6


def test_btlmd_dominant():
    W = np.array([[0, 10], [0, 0]], dtype=float)
    r = btlmd(W)
    assert r.value["rankings"][0] == 0


def test_btlmd_symmetric():
    W = np.array([[0, 5], [5, 0]], dtype=float)
    r = btlmd(W)
    assert abs(r.value["strengths"][0] - r.value["strengths"][1]) < 0.01
