"""Test comb_filter (cmbfl)."""
import numpy as np
from moirais.fn.cmbfl import comb_filter, cmbfl
from moirais.fn._containers import SignalResult


class TestCombFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter(x, delay=5)
        assert isinstance(result, SignalResult)
        assert result.name == "comb_filter"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = comb_filter(x, delay=10, alpha=0.3)
        assert result.n_samples == 256
        assert len(result.filtered) == 256

    def test_delay_effect(self):
        x = np.zeros(100)
        x[0] = 1.0
        result = comb_filter(x, delay=5, alpha=1.0)
        assert result.filtered[5] == 1.0

    def test_alias(self):
        assert cmbfl is comb_filter
