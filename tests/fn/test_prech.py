"""Tests for moirais.fn.prech — Proportional reduction in error."""
import numpy as np
from moirais.fn.prech import prech


def test_prech_perfect():
    V = np.array([[1, 0], [0, 1]], dtype=float)
    P = np.array([[1, 0], [0, 1]], dtype=float)
    r = prech(V, P)
    assert r.value == 1.0


def test_prech_null():
    V = np.array([[1, 1], [1, 1]], dtype=float)
    P = np.array([[1, 1], [1, 1]], dtype=float)
    r = prech(V, P)
    assert r.value >= 0


def test_prech_stats():
    V = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    P = np.array([[1, 0, 1], [0, 0, 0]], dtype=float)
    r = prech(V, P)
    assert r.extra["model_errors"] <= r.extra["null_errors"]
