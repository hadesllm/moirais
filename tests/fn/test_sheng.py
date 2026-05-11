"""Test shannon_energy_fn."""
import numpy as np
from morie.fn.sheng import shannon_energy_fn, alias
from morie.fn._containers import SignalResult


class TestShannonEnergyFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shannon_energy_fn(x)
        assert isinstance(result, SignalResult)

    def test_filtered_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shannon_energy_fn(x)
        assert result.filtered is not None
        assert len(result.filtered) == len(x)

    def test_n_samples(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shannon_energy_fn(x)
        assert result.n_samples == 256

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shannon_energy_fn(x)
        assert result.name == "shannon_energy"

    def test_alias(self):
        assert alias is shannon_energy_fn
