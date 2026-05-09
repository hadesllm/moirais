"""Tests for moirais.fn.spec — Specificity."""
import numpy as np

from moirais.fn.spec import specificity_dx, spec


def test_perfect_specificity():
    """All negatives correctly identified: specificity = 1."""
    y_true = [1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0]
    result = specificity_dx(y_true, y_pred)
    assert result.estimate == 1.0


def test_known_specificity():
    """TN=9, FP=1 -> specificity = 0.9."""
    y_true = [0] * 10 + [1] * 5
    y_pred = [0] * 9 + [1] * 1 + [1] * 5
    result = specificity_dx(y_true, y_pred)
    assert abs(result.estimate - 0.9) < 1e-10
    assert result.extra["tn"] == 9
    assert result.extra["fp"] == 1


def test_spec_alias():
    assert spec is specificity_dx
