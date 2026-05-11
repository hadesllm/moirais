"""Test signal_power (spowr)."""
import numpy as np
from morie.fn.spowr import signal_power, spowr
from morie.fn._containers import DescriptiveResult


class TestSignalPower:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = signal_power(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 14.0 / 3.0) < 1e-10

    def test_alias(self):
        assert spowr is signal_power
