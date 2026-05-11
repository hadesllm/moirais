"""Tests for morie.fn.sutrn -- substance use trend."""

import pytest
from morie.fn.sutrn import substance_trend


class TestSubstanceTrend:
    def test_increasing(self):
        res = substance_trend(rates=[0.10, 0.12, 0.15], periods=[2020, 2021, 2022])
        assert res.name == "substance_trend"
        assert res.value > 0

    def test_pct_change(self):
        res = substance_trend(rates=[0.10, 0.20], periods=[2020, 2021])
        assert res.extra["pct_change"] == pytest.approx(100.0)

    def test_short_raises(self):
        with pytest.raises(ValueError):
            substance_trend(rates=[0.1], periods=[2020])
