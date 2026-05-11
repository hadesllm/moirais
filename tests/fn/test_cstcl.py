"""Tests for morie.fn.cstcl — custody classification."""

import pytest
import numpy as np
from morie.fn.cstcl import custody_classification
from morie.fn._containers import DescriptiveResult


class TestCustodyClassification:

    def test_returns_descriptive(self):
        levels = np.array(["min", "med", "max", "min", "med"])
        result = custody_classification(levels)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_levels"] == 3

    def test_proportions_sum_one(self):
        levels = np.array([1, 2, 3, 1, 2, 3, 1, 1])
        result = custody_classification(levels)
        assert sum(result.extra["proportions"].values()) == pytest.approx(1.0)
