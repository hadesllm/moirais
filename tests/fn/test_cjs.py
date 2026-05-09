"""Tests for moirais.fn.cjs -- CJS flow calculation."""

import pytest
from moirais.fn.cjs import cjs_flow, cjs
from moirais.fn._containers import DescriptiveResult


class TestCjs:
    def test_alias(self):
        assert cjs is cjs_flow

    def test_basic_flow(self):
        result = cjs_flow(1000, 800, 600, 200)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["prosecution_rate"] == pytest.approx(0.8)
        assert result.extra["conviction_rate"] == pytest.approx(0.75)
        assert result.extra["incarceration_rate"] == pytest.approx(1 / 3, rel=1e-3)

    def test_overall_attrition(self):
        result = cjs_flow(1000, 800, 600, 200)
        assert result.extra["overall_attrition"] == pytest.approx(0.8)
