"""Test cross_psd (xpsd)."""
import numpy as np
from moirais.fn.xpsd import cross_psd, xpsd
from moirais.fn._containers import DescriptiveResult


class TestXpsd:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        y = rng.standard_normal(64)
        result = cross_psd(x, y, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cross_psd"

    def test_self_is_psd(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = cross_psd(x, x)
        assert np.all(np.real(result.extra["cpsd"]) >= -1e-10)

    def test_alias(self):
        assert xpsd is cross_psd
