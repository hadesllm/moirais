"""Tests for morie.fn.hrvmt -- HRV metrics."""

import numpy as np

from morie.fn.hrvmt import hrvmt


class TestHrvMt:
    def test_basic(self):
        rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.85, 0.77, 0.82])
        result = hrvmt(rr)
        assert result.name == "hrv_metrics"
        assert result.value > 0
        assert "sdnn" in result.extra
        assert "rmssd" in result.extra
        assert "pnn50" in result.extra
        assert "mean_hr" in result.extra

    def test_constant_rr(self):
        rr = np.ones(20) * 0.8
        result = hrvmt(rr)
        assert result.extra["rmssd"] == 0.0
        assert abs(result.extra["mean_hr"] - 75.0) < 0.1

    def test_short_sequence(self):
        rr = np.array([0.8])
        result = hrvmt(rr)
        assert result.extra["sdnn"] == 0
