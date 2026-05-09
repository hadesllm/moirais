"""Tests for moirais.fn.tpscl — clearance rate."""

import pytest
from moirais.fn.tpscl import tps_clearance_rate
from moirais.fn._containers import CrimeResult


class TestClearanceRate:
    def test_basic(self):
        r = tps_clearance_rate(80, 100)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.8)

    def test_ci_bounds(self):
        r = tps_clearance_rate(50, 100)
        assert r.ci_lower < 0.5 < r.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            tps_clearance_rate(10, 0)
