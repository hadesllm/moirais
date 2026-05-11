"""Tests for morie.fn.heboi -- burden of illness."""

import pytest
from morie.fn.heboi import burden_of_illness


class TestBurdenOfIllness:
    def test_basic(self):
        res = burden_of_illness(direct_costs=500, indirect_costs=300, intangible_costs=200)
        assert res.estimate == pytest.approx(1000.0)

    def test_pct(self):
        res = burden_of_illness(500, 500)
        assert res.extra["pct_direct"] == pytest.approx(50.0)
