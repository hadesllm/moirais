"""Test alpha_trimmed_mean_filter (atmfl)."""
import numpy as np
import pytest

from morie.fn.atmfl import alpha_trimmed_mean_filter, atmfl
from morie.fn._containers import SignalResult


class TestAlphaTrimmedMeanFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = alpha_trimmed_mean_filter(x)
        assert isinstance(result, SignalResult)
        assert result.name == "alpha_trimmed_mean_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = alpha_trimmed_mean_filter(x)
        assert result.n_samples == 256
        assert result.filtered is not None
        assert len(result.filtered) == 256

    def test_smoothing(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = alpha_trimmed_mean_filter(x, window=11)
        assert np.std(result.filtered) < np.std(x)

    def test_alias(self):
        assert atmfl is alpha_trimmed_mean_filter
