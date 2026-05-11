"""Test t_wave_detect (twave)."""
import numpy as np
import pytest

from morie.fn.twave import t_wave_detect, twave
from morie.fn._containers import DescriptiveResult


class TestTWaveDetect:
    def test_basic(self):
        rng = np.random.default_rng(42)
        ecg = rng.standard_normal(2000)
        ecg[500] = 5.0
        ecg[600] = 3.0
        qrs_locs = np.array([500])
        result = t_wave_detect(ecg, qrs_locs, fs=360.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "t_wave_detect"

    def test_extra_keys(self):
        ecg = np.random.default_rng(42).standard_normal(2000)
        qrs_locs = np.array([200, 600])
        result = t_wave_detect(ecg, qrs_locs, fs=360.0)
        assert "t_peak_indices" in result.extra

    def test_value_is_count(self):
        ecg = np.random.default_rng(42).standard_normal(2000)
        qrs_locs = np.array([200, 600])
        result = t_wave_detect(ecg, qrs_locs, fs=360.0)
        assert result.value == len(result.extra["t_peak_indices"])

    def test_alias(self):
        assert twave is t_wave_detect
