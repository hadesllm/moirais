"""Test roc_det_curve (rocdt)."""
import numpy as np
from morie.fn.rocdt import roc_det_curve, rocdt
from morie.fn._containers import DescriptiveResult


class TestRocdt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_scores = rng.uniform(0, 1, 10)
        result = roc_det_curve(y_true, y_scores)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "roc_det_curve"
        assert 0 <= result.extra["auc"] <= 1

    def test_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = roc_det_curve(y_true, y_scores)
        assert result.extra["auc"] == 1.0

    def test_alias(self):
        assert rocdt is roc_det_curve
