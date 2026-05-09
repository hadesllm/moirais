"""Test bartlett_psd (psdbt)."""
import numpy as np
from moirais.fn.psdbt import bartlett_psd, psdbt
from moirais.fn._containers import DescriptiveResult


class TestPsdbt:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = bartlett_psd(x, nseg=4, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bartlett_psd"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = bartlett_psd(x, nseg=8)
        assert np.all(result.extra["psd"] >= 0)

    def test_alias(self):
        assert psdbt is bartlett_psd
