"""Tests for morie.fn.gpopt -- GP Bayesian optimization."""

import numpy as np
from morie.fn.gpopt import gp_optimize


def _sphere(x):
    return float(np.sum(x ** 2))


def test_returns_dict():
    result = gp_optimize(_sphere, [[-5, 5]], n_calls=15, n_initial=5)
    assert isinstance(result, dict)
    assert "best_x" in result
    assert "best_y" in result


def test_finds_minimum():
    result = gp_optimize(_sphere, [[-5, 5]], n_calls=25, seed=42)
    assert result["best_y"] < 1.0


def test_history_shape():
    result = gp_optimize(_sphere, [[-5, 5], [-5, 5]], n_calls=15)
    assert result["x_history"].shape == (15, 2)
    assert len(result["y_history"]) == 15


def test_best_in_history():
    result = gp_optimize(_sphere, [[-5, 5]], n_calls=15)
    assert result["best_y"] == np.min(result["y_history"])
