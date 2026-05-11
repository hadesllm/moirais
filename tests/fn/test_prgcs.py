"""Tests for morie.fn.prgcs — program cost savings."""

import pytest
from morie.fn.prgcs import program_cost_savings
from morie.fn._containers import ESRes


class TestProgramCostSavings:
    def test_positive_net(self):
        r = program_cost_savings(0.1, 100000, 500, 1000000)
        assert isinstance(r, ESRes)
        assert r.estimate > 0
        assert r.extra["avoided_recid"] == pytest.approx(50.0)

    def test_negative_net(self):
        r = program_cost_savings(0.01, 10000, 10, 1000000)
        assert r.estimate < 0
