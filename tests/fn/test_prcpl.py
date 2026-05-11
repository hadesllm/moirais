"""Test precision_recall_curve (prcpl)."""
import numpy as np
from morie.fn.prcpl import precision_recall_curve, prcpl
from morie.fn._containers import DescriptiveResult


class TestPrcpl:
    def test_basic(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        result = precision_recall_curve(y_true, y_scores)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "precision_recall_curve"
        assert 0 <= result.extra["average_precision"] <= 1

    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = precision_recall_curve(y_true, y_scores)
        assert result.extra["average_precision"] == 1.0

    def test_alias(self):
        assert prcpl is precision_recall_curve
