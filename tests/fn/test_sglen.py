"""Test signal_arc_length."""
import numpy as np
from moirais.fn.sglen import signal_arc_length, sglen
from moirais.fn._containers import DescriptiveResult


class TestSignalArcLength:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = signal_arc_length(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_positive(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = signal_arc_length(x)
        assert isinstance(result.value, float)
        assert result.value > 0.0

    def test_constant_signal(self):
        x = np.ones(100)
        result = signal_arc_length(x)
        assert abs(result.value - 99.0) < 1e-6

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = signal_arc_length(x)
        assert result.name == "signal_arc_length"

    def test_alias(self):
        assert sglen is signal_arc_length
