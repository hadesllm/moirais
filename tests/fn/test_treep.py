"""Tests for morie.fn.treep -- TPE optimizer."""

import numpy as np
from morie.fn.treep import tpe_minimize


def _sphere(x):
    return float(np.sum(x ** 2))


def test_returns_dict():
    result = tpe_minimize(_sphere, [[- 5, 5]], n_calls=20, n_initial=5)
    assert isinstance(result, dict)
    assert "best_x" in result
    assert "best_y" in result


def test_finds_minimum():
    result = tpe_minimize(_sphere, [[-5, 5]], n_calls=50, seed=42)
    assert result["best_y"] < 1.0


def test_history_shape():
    result = tpe_minimize(_sphere, [[-5, 5], [-5, 5]], n_calls=30)
    assert result["x_history"].shape == (30, 2)
    assert len(result["y_history"]) == 30


def test_best_in_history():
    result = tpe_minimize(_sphere, [[-5, 5]], n_calls=30)
    assert result["best_y"] == np.min(result["y_history"])


def test_multidimensional():
    result = tpe_minimize(_sphere, [[-5, 5], [-5, 5], [-5, 5]], n_calls=60, seed=42)
    assert result["best_y"] < 5.0
