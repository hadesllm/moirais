"""Test derivative_hz (drvhz)."""
import numpy as np

from moirais.fn.drvhz import derivative_hz, drvhz
from moirais.fn._containers import DescriptiveResult


class TestDerivativeHz:
    def test_basic(self):
        t = np.linspace(0, 1, 256)
        x = np.sin(2 * np.pi * 5 * t)
        result = derivative_hz(x, fs=256.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "derivative_hz"
        assert result.value > 0

    def test_constant_signal(self):
        x = np.ones(64)
        result = derivative_hz(x, fs=1.0)
        assert result.value < 1e-10

    def test_alias(self):
        assert drvhz is derivative_hz
