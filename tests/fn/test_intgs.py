"""Test integrate_signal (intgs)."""
import numpy as np
from moirais.fn.intgs import integrate_signal, intgs
from moirais.fn._containers import SignalResult


class TestIntegrateSignal:
    def test_basic(self):
        x = np.ones(100)
        result = integrate_signal(x, fs=1.0)
        assert isinstance(result, SignalResult)
        assert result.name == "integrate_signal"

    def test_constant(self):
        x = np.ones(10)
        result = integrate_signal(x, fs=1.0)
        assert abs(result.filtered[-1] - 9.0) < 1e-10

    def test_alias(self):
        assert intgs is integrate_signal
