"""Tests for morie.fn.youden — Youden's J index."""
import numpy as np

from morie.fn.youden import youdens_j, youden


def test_perfect_youden():
    """Perfect separation: J = 1.0."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = youdens_j(y_true, y_score)
    assert abs(result.estimate - 1.0) < 1e-10


def test_optimal_threshold():
    """Optimal threshold should be between the groups."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = youdens_j(y_true, y_score)
    assert 0.3 <= result.extra["optimal_threshold"] <= 0.7


def test_youden_alias():
    assert youden is youdens_j
