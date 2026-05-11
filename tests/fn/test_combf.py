"""Test comb_filter_signal (combf)."""
import numpy as np
import pytest

from morie.fn.combf import comb_filter_signal, combf
from morie.fn._containers import SignalResult


class TestCombFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter_signal(x, fundamental=60.0, fs=1000.0)
        assert isinstance(result, SignalResult)
        assert result.name == "comb_filter_signal"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter_signal(x, fundamental=60.0, fs=1000.0)
        assert result.n_samples == 256

    def test_fs_stored(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter_signal(x, fundamental=60.0, fs=1000.0)
        assert result.fs == 1000.0

    def test_harmonics_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter_signal(x, fundamental=50.0, fs=500.0, n_harmonics=3, q=20.0)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert combf is comb_filter_signal
