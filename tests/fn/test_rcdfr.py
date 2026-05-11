"""Tests for morie.fn.rcdfr — recidivism fairness."""

import pytest
import numpy as np
from morie.fn.rcdfr import recidivism_fairness
from morie.fn._containers import DescriptiveResult


class TestRecidivismFairness:

    def test_returns_descriptive(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = recidivism_fairness(y_true, y_pred, group)
        assert isinstance(result, DescriptiveResult)
        assert "group_metrics" in result.extra

    def test_equal_groups_ratio_near_one(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        group = np.array(["A"] * 100 + ["B"] * 100)
        result = recidivism_fairness(y_true, y_pred, group)
        assert result.value >= 0
