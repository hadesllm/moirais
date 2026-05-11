"""Test hann_filter (hannf)."""
import numpy as np
import pytest

from morie.fn.hannf import hann_filter, hannf
from morie.fn._containers import SignalResult


class TestHannFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hann_filter(x)
        assert isinstance(result, SignalResult)
        assert result.name == "hann_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hann_filter(x)
        assert result.n_samples == 256
        assert result.filtered is not None
        assert len(result.filtered) == 256

    def test_smoothing(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hann_filter(x, window=11)
        assert np.std(result.filtered) < np.std(x)

    def test_alias(self):
        assert hannf is hann_filter
