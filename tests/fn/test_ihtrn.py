"""Tests for morie.fn.ihtrn -- indigenous health trend."""

from morie.fn.ihtrn import indigenous_health_trend


class TestIndigenousHealthTrend:
    def test_improving(self):
        res = indigenous_health_trend(rates=[0.20, 0.18, 0.15], years=[2020, 2021, 2022])
        assert res.value < 0
        assert res.extra["improving"] is True

    def test_worsening(self):
        res = indigenous_health_trend(rates=[0.10, 0.12, 0.15], years=[2020, 2021, 2022])
        assert res.value > 0
