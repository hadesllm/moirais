"""Test snr_compute (ssnr)."""
import numpy as np
from morie.fn.ssnr import snr_compute, ssnr
from morie.fn._containers import DescriptiveResult


class TestSNR:
    def test_equal_power(self):
        s = np.array([1.0, -1.0, 1.0, -1.0])
        n = np.array([1.0, -1.0, 1.0, -1.0])
        result = snr_compute(s, n)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 0.0) < 1e-10

    def test_zero_noise(self):
        s = np.array([1.0, 2.0])
        n = np.zeros(2)
        assert snr_compute(s, n).value == float("inf")

    def test_alias(self):
        assert ssnr is snr_compute
