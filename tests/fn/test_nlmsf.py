"""Test nlms_filter (nlmsf)."""
import numpy as np
from moirais.fn.nlmsf import nlms_filter, nlmsf
from moirais.fn._containers import SignalResult


class TestNlmsFilter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        d = np.convolve(x, [1.0, 0.5], mode="full")[:500]
        result = nlms_filter(x, d, mu=0.5, order=8)
        assert isinstance(result, SignalResult)
        assert result.name == "nlms_filter"

    def test_output_length(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        d = rng.standard_normal(200)
        result = nlms_filter(x, d, order=4)
        assert result.n_samples == 200

    def test_has_weights(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        d = rng.standard_normal(200)
        result = nlms_filter(x, d, order=8)
        assert "weights" in result.extra
        assert len(result.extra["weights"]) == 8

    def test_alias(self):
        assert nlmsf is nlms_filter
