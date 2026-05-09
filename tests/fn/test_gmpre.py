"""Tests for moirais.fn.gmpre — Geometric mean probability."""
import numpy as np
from moirais.fn.gmpre import gmpre


def test_gmpre_perfect():
    V = np.array([[1, 0], [0, 1]], dtype=float)
    P = np.array([[0.99, 0.01], [0.01, 0.99]])
    r = gmpre(V, P)
    assert r.value > 0.95


def test_gmpre_random():
    V = np.array([[1, 0], [0, 1]], dtype=float)
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    r = gmpre(V, P)
    assert abs(r.value - 0.5) < 0.01


def test_gmpre_per_leg():
    V = np.array([[1, 1], [0, 0]], dtype=float)
    P = np.array([[0.9, 0.9], [0.1, 0.1]])
    r = gmpre(V, P)
    assert len(r.extra["per_legislator_gmp"]) == 2
