"""Test noise_power (npowr)."""
import numpy as np
from morie.fn.npowr import noise_power, npowr
from morie.fn._containers import DescriptiveResult


class TestNoisePower:
    def test_with_signal(self):
        s = np.array([1.0, 2.0, 3.0])
        x = np.array([1.5, 2.5, 3.5])
        result = noise_power(x, signal=s)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 0.25) < 1e-10

    def test_without_signal(self):
        x = np.array([1.0, 1.0, 1.0])
        assert noise_power(x).value == 0.0

    def test_alias(self):
        assert npowr is noise_power
