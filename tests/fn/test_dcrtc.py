"""Test dicrotic_notch_detect (dcrtc)."""
import numpy as np
import pytest

from moirais.fn.dcrtc import dicrotic_notch_detect, dcrtc
from moirais.fn._containers import DescriptiveResult


class TestDicroticNotchDetect:
    def test_basic(self):
        t = np.arange(0, 2, 1/125)
        pulse = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 3.0 * t)
        result = dicrotic_notch_detect(pulse, fs=125.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "dicrotic_notch"

    def test_extra_keys(self):
        pulse = np.random.default_rng(42).standard_normal(500)
        result = dicrotic_notch_detect(pulse, fs=125.0)
        assert "notch_indices" in result.extra

    def test_value_is_count(self):
        pulse = np.random.default_rng(42).standard_normal(500)
        result = dicrotic_notch_detect(pulse, fs=125.0)
        assert result.value == len(result.extra["notch_indices"])

    def test_alias(self):
        assert dcrtc is dicrotic_notch_detect
