"""Tests for moirais.fn.recid -- Recidivism rate."""

import pytest
from moirais.fn.recid import recidivism_rate, recid
from moirais.fn._containers import CrimeResult


class TestRecid:
    def test_alias(self):
        assert recid is recidivism_rate

    def test_basic_rate(self):
        result = recidivism_rate(30, 100)
        assert isinstance(result, CrimeResult)
        assert result.rate == pytest.approx(0.3)

    def test_ci_bounds(self):
        result = recidivism_rate(30, 100)
        assert 0.0 <= result.ci_lower <= result.rate
        assert result.rate <= result.ci_upper <= 1.0
