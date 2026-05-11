"""Test wiener_filter (wienr)."""
import numpy as np
import pytest

from morie.fn.wienr import wiener_filter, wienr
from morie.fn._containers import SignalResult


class TestWienerFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = wiener_filter(x)
        assert isinstance(result, SignalResult)
        assert result.name == "wiener_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = wiener_filter(x)
        assert result.n_samples == 256

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = wiener_filter(x)
        assert result.filtered is not None

    def test_noise_fraction_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = wiener_filter(x, noise_fraction=0.2)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert wienr is wiener_filter
