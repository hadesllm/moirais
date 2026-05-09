"""Tests for moirais.fn.cstfr — custody facility rate."""

import pytest
import numpy as np
from moirais.fn.cstfr import custody_facility_rate
from moirais.fn._containers import DescriptiveResult


class TestCustodyFacilityRate:

    def test_returns_descriptive(self):
        counts = np.array([5, 10, 15])
        pops = np.array([100, 200, 300])
        fids = np.array(["F1", "F2", "F3"])
        result = custody_facility_rate(counts, pops, fids)
        assert isinstance(result, DescriptiveResult)

    def test_rate_calculation(self):
        result = custody_facility_rate(np.array([10]), np.array([1000]), np.array(["X"]))
        assert result.value == pytest.approx(10.0)
