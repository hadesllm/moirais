"""Test noise_psd (npsd)."""
import numpy as np
from moirais.fn.npsd import noise_psd, npsd
from moirais.fn._containers import DescriptiveResult


class TestNoisePSD:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        result = noise_psd(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_alias(self):
        assert npsd is noise_psd
