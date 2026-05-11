"""Tests for morie.fn.henb -- net monetary benefit."""

import pytest
from morie.fn.henb import net_monetary_benefit


class TestNMB:
    def test_positive(self):
        res = net_monetary_benefit(effect_diff=2.0, cost_diff=10000, wtp=50000)
        assert res.estimate == pytest.approx(90000.0)
        assert res.extra["cost_effective"] is True

    def test_negative(self):
        res = net_monetary_benefit(effect_diff=0.1, cost_diff=50000, wtp=50000)
        assert res.extra["cost_effective"] is False
