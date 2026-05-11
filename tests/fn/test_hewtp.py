"""Tests for morie.fn.hewtp -- willingness to pay."""

import pytest
from morie.fn.hewtp import willingness_to_pay


class TestWTP:
    def test_cost_effective(self):
        res = willingness_to_pay(effect=2.0, cost=50000, wtp_per_qaly=50000)
        assert res.estimate == pytest.approx(25000.0)
        assert res.extra["cost_effective"] is True

    def test_not_ce(self):
        res = willingness_to_pay(effect=0.5, cost=50000, wtp_per_qaly=50000)
        assert res.extra["cost_effective"] is False
