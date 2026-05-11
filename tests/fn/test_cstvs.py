"""Tests for morie.fn.cstvs — custody visits."""

import pytest
import numpy as np
from morie.fn.cstvs import custody_visits
from morie.fn._containers import DescriptiveResult


class TestCustodyVisits:

    def test_returns_descriptive(self):
        vc = np.array([0, 1, 2, 3, 0, 5])
        result = custody_visits(vc)
        assert isinstance(result, DescriptiveResult)

    def test_all_zero(self):
        result = custody_visits(np.array([0, 0, 0]))
        assert result.extra["pct_zero_visits"] == pytest.approx(1.0)
