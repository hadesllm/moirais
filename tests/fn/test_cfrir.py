"""Tests for morie.fn.cfrir -- case fatality rate."""

import pytest
from morie.fn.cfrir import case_fatality_rate


class TestCFR:
    def test_naive_cfr(self):
        res = case_fatality_rate(10, 100)
        assert res["cfr_naive"] == pytest.approx(0.10)

    def test_corrected_cfr(self):
        res = case_fatality_rate(10, 100, resolved=80)
        assert res["cfr_corrected"] == pytest.approx(10 / 80)

    def test_ci_contains_cfr(self):
        res = case_fatality_rate(50, 500)
        assert res["ci_lower"] < 0.10 < res["ci_upper"]

    def test_zero_deaths(self):
        res = case_fatality_rate(0, 100)
        assert res["cfr_naive"] == 0.0
        assert res["ci_lower"] == 0.0

    def test_resolved_less_than_deaths_raises(self):
        with pytest.raises(ValueError):
            case_fatality_rate(10, 100, resolved=5)

    def test_deaths_exceed_cases_raises(self):
        with pytest.raises(ValueError):
            case_fatality_rate(101, 100)
