"""Test power_cepstrum (prcep)."""
import numpy as np
from morie.fn.prcep import power_cepstrum, prcep
from morie.fn._containers import DescriptiveResult


class TestPrcep:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = power_cepstrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "power_cepstrum"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = power_cepstrum(x)
        assert np.all(result.extra["cepstrum"] >= 0)

    def test_alias(self):
        assert prcep is power_cepstrum
