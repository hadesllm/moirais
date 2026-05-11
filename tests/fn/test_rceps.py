"""Test real_cepstrum (rceps)."""
import numpy as np
from morie.fn.rceps import real_cepstrum, rceps
from morie.fn._containers import DescriptiveResult


class TestRceps:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = real_cepstrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "real_cepstrum"

    def test_finite(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = real_cepstrum(x)
        assert np.all(np.isfinite(result.extra["cepstrum"]))

    def test_alias(self):
        assert rceps is real_cepstrum
