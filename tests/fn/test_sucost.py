"""Tests for morie.fn.sucost -- substance cost."""

import pytest
from morie.fn.sucost import substance_cost


class TestSubstanceCost:
    def test_basic(self):
        res = substance_cost(medical_costs=100, productivity_costs=200, criminal_costs=50)
        assert res.estimate == pytest.approx(350.0)

    def test_pct(self):
        res = substance_cost(medical_costs=500, productivity_costs=500)
        assert res.extra["pct_medical"] == pytest.approx(50.0)
