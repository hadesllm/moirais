"""Test power_spectrum (pwrsp)."""
import numpy as np
from moirais.fn.pwrsp import power_spectrum, pwrsp
from moirais.fn._containers import DescriptiveResult


class TestPwrsp:
    def test_basic(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        result = power_spectrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "power_spectrum"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = power_spectrum(x)
        assert np.all(result.extra["power"] >= 0)

    def test_alias(self):
        assert pwrsp is power_spectrum
