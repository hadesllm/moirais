"""Tests for morie.fn.tpscb — cost benefit."""

import pytest
from morie.fn.tpscb import tps_cost_benefit
from morie.fn._containers import ESRes


class TestCostBenefit:
    def test_positive_bcr(self):
        r = tps_cost_benefit(100000, 300000)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(3.0, rel=0.01)

    def test_net_benefit(self):
        r = tps_cost_benefit(100000, 200000, years=3)
        assert r.extra["net_benefit"] > 0

    def test_invalid(self):
        with pytest.raises(ValueError):
            tps_cost_benefit(0, 100)
