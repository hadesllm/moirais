"""Test differentiate_signal (diffs)."""
import numpy as np
from morie.fn.diffs import differentiate_signal, diffs
from morie.fn._containers import SignalResult


class TestDifferentiateSignal:
    def test_basic(self):
        x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        result = differentiate_signal(x)
        assert isinstance(result, SignalResult)
        assert result.name == "differentiate_signal"

    def test_linear(self):
        x = np.arange(10, dtype=float)
        result = differentiate_signal(x, fs=1.0)
        np.testing.assert_allclose(result.filtered, np.ones(10), atol=1e-10)

    def test_alias(self):
        assert diffs is differentiate_signal
