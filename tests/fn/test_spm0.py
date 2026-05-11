"""Test spectral_moment (spm0)."""
import numpy as np
from morie.fn.spm0 import spectral_moment, spm0
from morie.fn._containers import DescriptiveResult


class TestSpm0:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_moment(x, fs=100.0, k=0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_moment"

    def test_zeroth_positive(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_moment(x, k=0)
        assert result.value > 0

    def test_alias(self):
        assert spm0 is spectral_moment
