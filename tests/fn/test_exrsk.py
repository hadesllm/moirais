"""Tests for morie.fn.exrsk — Excess risk."""

import pytest

from morie.fn.exrsk import excess_risk


class TestExcessRisk:
    def test_basic(self):
        res = excess_risk(0.30, 0.10)
        assert res.estimate == pytest.approx(0.20)

    def test_with_ci(self):
        res = excess_risk(0.30, 0.10, n_exposed=200, n_unexposed=200)
        assert res.ci_lower is not None
        assert res.ci_lower < 0.20 < res.ci_upper

    def test_no_difference(self):
        res = excess_risk(0.10, 0.10)
        assert res.estimate == pytest.approx(0.0)
