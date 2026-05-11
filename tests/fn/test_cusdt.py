"""Test cusum_detect (cusdt)."""
import numpy as np
from morie.fn.cusdt import cusum_detect, cusdt
from morie.fn._containers import DescriptiveResult


class TestCusumDetect:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 3])
        result = cusum_detect(x, threshold=5.0, drift=0.5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cusum_detect"

    def test_detects_shift(self):
        x = np.concatenate([np.zeros(100), np.ones(100) * 5])
        result = cusum_detect(x, threshold=3.0, drift=0.5)
        assert result.value > 0

    def test_no_alarm_stable(self):
        x = np.zeros(200)
        result = cusum_detect(x, threshold=5.0, drift=0.5)
        assert result.value == 0

    def test_alias(self):
        assert cusdt is cusum_detect
