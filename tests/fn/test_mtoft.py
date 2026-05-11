"""Tests for morie.fn.mtoft — fatality rate."""

import pytest
from morie.fn.mtoft import mto_fatality_rate
from morie.fn._containers import CrimeResult


class TestFatalityRate:
    def test_basic(self):
        r = mto_fatality_rate(150, 15000000)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(1.0)

    def test_ci(self):
        r = mto_fatality_rate(50, 1000000)
        assert r.ci_lower < r.rate < r.ci_upper
