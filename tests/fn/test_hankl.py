"""Tests for hankl.py - Hankel matrix."""
import numpy as np
from morie.fn.hankl import hankel_matrix_fn, hankl


def test_hankl_returns_result():
    x = np.arange(10, dtype=float)
    result = hankel_matrix_fn(x, L=4)
    assert result.name == "hankel_matrix"
    assert result.extra["shape"] == (4, 7)


def test_hankl_matrix_values():
    x = np.arange(6, dtype=float)
    result = hankel_matrix_fn(x, L=3)
    H = result.extra["matrix"]
    assert H[0, 0] == 0.0
    assert H[2, 0] == 2.0
    assert H[0, 3] == 3.0


def test_hankl_alias():
    x = np.arange(8, dtype=float)
    result = hankl(x, L=3)
    assert result.name == "hankel_matrix"
