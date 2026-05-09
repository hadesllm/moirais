"""Test thd_compute (thd)."""
import numpy as np
from moirais.fn.thd import thd_compute, thd
from moirais.fn._containers import DescriptiveResult


class TestThd:
    def test_basic(self):
        t = np.arange(1024) / 1024.0
        x = np.sin(2 * np.pi * 50 * t)
        result = thd_compute(x, fs=1024.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "thd"

    def test_pure_tone_low_thd(self):
        t = np.arange(1024) / 1024.0
        x = np.sin(2 * np.pi * 50 * t)
        result = thd_compute(x, fs=1024.0)
        assert result.value < 0.1

    def test_alias(self):
        assert thd is thd_compute
