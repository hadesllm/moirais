"""Tests for moirais.fn.mhwat -- wait time analysis."""

import pytest
from moirais.fn.mhwat import wait_time_analysis


class TestWaitTime:
    def test_basic(self):
        res = wait_time_analysis([7, 14, 21, 35, 60])
        assert res.name == "wait_time_analysis"
        assert res.value == pytest.approx(21.0)

    def test_benchmark(self):
        res = wait_time_analysis([10, 20, 40, 50], benchmark_days=30)
        assert res.extra["pct_over_benchmark"] == pytest.approx(50.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            wait_time_analysis([])
