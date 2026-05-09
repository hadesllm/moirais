"""Test trim_signal (sgtrm)."""
import numpy as np
from moirais.fn.sgtrm import trim_signal, sgtrm
from moirais.fn._containers import SignalResult


class TestTrimSignal:
    def test_basic(self):
        x = np.arange(100, dtype=float)
        result = trim_signal(x, start=10, end=50)
        assert isinstance(result, SignalResult)
        assert result.n_samples == 40

    def test_values(self):
        x = np.arange(10, dtype=float)
        result = trim_signal(x, start=2, end=5)
        np.testing.assert_array_equal(result.filtered, [2, 3, 4])

    def test_alias(self):
        assert sgtrm is trim_signal
