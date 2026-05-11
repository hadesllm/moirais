"""Tests for morie.fn.iv_wk -- Weak instrument diagnostics."""

import pytest
from morie.fn.iv_wk import weak_instrument_test


class TestWeakInstrument:
    def test_strong_instrument(self):
        result = weak_instrument_test(25.0, n_instruments=2)
        assert result["is_weak"] is False
        assert result["passes_stock_yogo"] is True

    def test_weak_instrument(self):
        result = weak_instrument_test(5.0, n_instruments=2)
        assert result["is_weak"] is True
        assert result["passes_stock_yogo"] is False

    def test_just_identified_no_sy(self):
        """Single instrument: Stock-Yogo CV not available."""
        result = weak_instrument_test(15.0, n_instruments=1)
        assert result["stock_yogo_5pct"] is None
        assert result["passes_stock_yogo"] is None

    def test_threshold_at_10(self):
        result = weak_instrument_test(10.0)
        assert result["is_weak"] is False  # not strictly less than 10
