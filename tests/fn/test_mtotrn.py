"""Tests for moirais.fn.mtotrn — road safety trend."""

import pytest
from moirais.fn.mtotrn import mto_trend
from moirais.fn._containers import DescriptiveResult


class TestMtoTrend:
    def test_improving(self):
        r = mto_trend([100, 90, 80, 70, 60])
        assert r.extra["trend"] == "improving"

    def test_too_short(self):
        with pytest.raises(ValueError):
            mto_trend([10, 20])
