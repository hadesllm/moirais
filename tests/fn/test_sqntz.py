"""Test quantize_signal (sqntz)."""
import numpy as np
from moirais.fn.sqntz import quantize_signal, sqntz
from moirais.fn._containers import SignalResult


class TestQuantizeSignal:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(100)
        result = quantize_signal(x, bits=8)
        assert isinstance(result, SignalResult)
        assert result.name == "quantize_signal"

    def test_levels(self):
        x = np.linspace(-1, 1, 1000)
        result = quantize_signal(x, bits=4)
        unique_vals = np.unique(result.filtered)
        assert len(unique_vals) <= 16

    def test_alias(self):
        assert sqntz is quantize_signal
