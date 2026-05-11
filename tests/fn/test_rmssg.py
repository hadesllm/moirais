"""Test rms_signal."""
import numpy as np
from morie.fn.rmssg import rms_signal, alias
from morie.fn._containers import DescriptiveResult


class TestRmsSignal:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = rms_signal(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_positive(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = rms_signal(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_known_value(self):
        x = np.ones(256)
        result = rms_signal(x)
        assert abs(result.value - 1.0) < 1e-9

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = rms_signal(x)
        assert result.name == "rms"

    def test_alias(self):
        assert alias is rms_signal
