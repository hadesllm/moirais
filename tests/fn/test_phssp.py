"""Test phase_spectrum (phssp)."""
import numpy as np
from moirais.fn.phssp import phase_spectrum, phssp
from moirais.fn._containers import DescriptiveResult


class TestPhssp:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = phase_spectrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "phase_spectrum"

    def test_range(self):
        x = np.random.default_rng(42).standard_normal(32)
        result = phase_spectrum(x)
        assert np.all(np.abs(result.extra["phase"]) <= np.pi + 1e-10)

    def test_alias(self):
        assert phssp is phase_spectrum
