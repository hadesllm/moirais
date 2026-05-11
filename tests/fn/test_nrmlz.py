"""Test normalize_signal (nrmlz)."""
import numpy as np
from morie.fn.nrmlz import normalize_signal, nrmlz
from morie.fn._containers import SignalResult


class TestNormalizeSignal:
    def test_zscore(self):
        x = np.random.default_rng(42).standard_normal(100) * 5 + 10
        result = normalize_signal(x, method="zscore")
        assert isinstance(result, SignalResult)
        assert abs(np.mean(result.filtered)) < 1e-10
        assert abs(np.std(result.filtered) - 1.0) < 1e-10

    def test_minmax(self):
        x = np.random.default_rng(42).standard_normal(100)
        result = normalize_signal(x, method="minmax")
        assert abs(np.min(result.filtered)) < 1e-10
        assert abs(np.max(result.filtered) - 1.0) < 1e-10

    def test_energy(self):
        x = np.array([3.0, 4.0])
        result = normalize_signal(x, method="energy")
        assert abs(np.sum(result.filtered ** 2) - 1.0) < 1e-10

    def test_alias(self):
        assert nrmlz is normalize_signal
