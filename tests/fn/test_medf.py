"""Test median_filter_signal (medf)."""
import numpy as np
import pytest

from moirais.fn.medf import median_filter_signal, medf
from moirais.fn._containers import SignalResult


class TestMedianFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = median_filter_signal(x)
        assert isinstance(result, SignalResult)
        assert result.name == "median_filter_signal"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = median_filter_signal(x)
        assert result.n_samples == 256

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = median_filter_signal(x)
        assert result.filtered is not None

    def test_kernel_size_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = median_filter_signal(x, kernel_size=7)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert medf is median_filter_signal
