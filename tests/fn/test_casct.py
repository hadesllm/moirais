"""Tests for moirais.fn.casct -- Case-control odds ratio."""

import pytest
from moirais.fn.casct import case_control_or


class TestCaseControlOR:
    def test_known(self):
        res = case_control_or(a=20, b=80, c=10, d=90)
        assert res.measure == "OR_cc"
        assert res.estimate == pytest.approx(20 * 90 / (80 * 10))

    def test_ci(self):
        res = case_control_or(a=20, b=80, c=10, d=90)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_zero_cell(self):
        with pytest.raises(ValueError):
            case_control_or(a=0, b=10, c=10, d=10)
