"""Tests for irteq -- IRT true-score equating."""
import numpy as np
from morie.fn.irteq import irt_equating
from morie.fn._containers import DescriptiveResult


class TestIrtEquating:
    def test_basic(self):
        a = {f"item_{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        b = {f"item_{j}": {"a": 1.2, "b": float(j - 1)} for j in range(5)}
        result = irt_equating(a, b)
        assert isinstance(result, DescriptiveResult)
        assert "true_score_a" in result.value

    def test_scores_monotonic(self):
        a = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(3)}
        b = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(3)}
        result = irt_equating(a, b)
        ts_a = np.array(result.value["true_score_a"])
        assert ts_a[-1] > ts_a[0]
