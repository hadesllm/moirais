"""Test log_magnitude_spectrum (logmg)."""
import numpy as np
from moirais.fn.logmg import log_magnitude_spectrum, logmg
from moirais.fn._containers import DescriptiveResult


class TestLogmg:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = log_magnitude_spectrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "log_magnitude_spectrum"

    def test_finite(self):
        x = np.random.default_rng(42).standard_normal(32)
        result = log_magnitude_spectrum(x)
        assert np.all(np.isfinite(result.extra["log_magnitude_db"]))

    def test_alias(self):
        assert logmg is log_magnitude_spectrum
