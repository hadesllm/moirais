"""Tests for morie.fn.phrx — prescription pattern analysis."""
import numpy as np
import pytest
from morie.fn.phrx import prescription_patterns


class TestPrescriptionPatterns:
    def test_basic_adherence(self):
        rx_dates = np.array([0, 30, 60, 90, 120], dtype=float)
        rx_durations = np.array([30, 30, 30, 30, 30], dtype=float)
        res = prescription_patterns(rx_dates, rx_durations)
        assert 0 <= res.extra["pdc"] <= 1
        assert 0 <= res.extra["mpr"] <= 1

    def test_perfect_adherence(self):
        rx_dates = np.array([0, 30, 60], dtype=float)
        rx_durations = np.array([30, 30, 30], dtype=float)
        res = prescription_patterns(rx_dates, rx_durations)
        assert res.extra["pdc"] > 0.9
        assert res.extra["mpr"] > 0.9

    def test_gaps_detected(self):
        rx_dates = np.array([0, 90], dtype=float)
        rx_durations = np.array([30, 30], dtype=float)
        res = prescription_patterns(rx_dates, rx_durations)
        assert res.extra["pdc"] < 0.7
