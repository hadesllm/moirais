"""Test pad_signal (sgpad)."""
import numpy as np
from morie.fn.sgpad import pad_signal, sgpad
from morie.fn._containers import SignalResult


class TestPadSignal:
    def test_zero_pad(self):
        x = np.array([1.0, 2.0, 3.0])
        result = pad_signal(x, pad_len=2, mode="zero")
        assert isinstance(result, SignalResult)
        assert result.n_samples == 7
        assert result.filtered[0] == 0.0
        assert result.filtered[-1] == 0.0

    def test_mirror(self):
        x = np.array([1.0, 2.0, 3.0])
        result = pad_signal(x, pad_len=2, mode="mirror")
        assert result.n_samples == 7

    def test_alias(self):
        assert sgpad is pad_signal
