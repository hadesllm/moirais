"""Tests for morie.fn.sens — Sensitivity."""
import numpy as np

from morie.fn.sens import sensitivity_dx, sens


def test_perfect_sensitivity():
    """All positives correctly identified: sensitivity = 1."""
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 0, 0]
    result = sensitivity_dx(y_true, y_pred)
    assert result.estimate == 1.0


def test_known_sensitivity():
    """TP=8, FN=2 -> sensitivity = 0.8."""
    y_true = [1] * 10 + [0] * 5
    y_pred = [1] * 8 + [0] * 2 + [0] * 5
    result = sensitivity_dx(y_true, y_pred)
    assert abs(result.estimate - 0.8) < 1e-10
    assert result.extra["tp"] == 8
    assert result.extra["fn"] == 2


def test_sens_alias():
    assert sens is sensitivity_dx
