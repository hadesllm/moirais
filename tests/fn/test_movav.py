"""Test moving_average (movav)."""
import numpy as np
import pytest

from morie.fn.movav import moving_average, movav
from morie.fn._containers import SignalResult


class TestMovingAverage:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = moving_average(x)
        assert isinstance(result, SignalResult)
        assert result.name == "moving_average"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = moving_average(x)
        assert result.n_samples == 256

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = moving_average(x)
        assert result.filtered is not None

    def test_window_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = moving_average(x, window=10)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert movav is moving_average
