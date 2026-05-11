"""Test sampling_theorem_check (smpth)."""
import numpy as np
from morie.fn.smpth import sampling_theorem_check, smpth
from morie.fn._containers import DescriptiveResult


class TestSamplingTheoremCheck:
    def test_satisfied(self):
        x = np.zeros(100)
        result = sampling_theorem_check(x, fs=1000.0, fmax=400.0)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0
        assert result.extra["satisfied"] is True

    def test_not_satisfied(self):
        x = np.zeros(100)
        result = sampling_theorem_check(x, fs=500.0, fmax=400.0)
        assert result.value == 0.0
        assert result.extra["satisfied"] is False

    def test_alias(self):
        assert smpth is sampling_theorem_check
