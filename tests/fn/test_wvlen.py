"""Test waveform_length_fn."""
import numpy as np
from morie.fn.wvlen import waveform_length_fn, alias
from morie.fn._containers import DescriptiveResult


class TestWaveformLengthFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = waveform_length_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_positive(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = waveform_length_fn(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_constant_signal_zero_length(self):
        x = np.ones(256)
        result = waveform_length_fn(x)
        assert result.value == 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = waveform_length_fn(x)
        assert result.name == "waveform_length"

    def test_alias(self):
        assert alias is waveform_length_fn
