"""Test equal_error_rate (eercl)."""
import numpy as np
from moirais.fn.eercl import equal_error_rate, eercl
from moirais.fn._containers import DescriptiveResult


class TestEercl:
    def test_basic(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
        result = equal_error_rate(y_true, y_scores)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "equal_error_rate"
        assert 0 <= result.extra["eer"] <= 1

    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        result = equal_error_rate(y_true, y_scores)
        assert result.extra["eer"] < 0.2

    def test_alias(self):
        assert eercl is equal_error_rate
