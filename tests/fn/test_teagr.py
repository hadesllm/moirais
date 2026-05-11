"""Test teager_energy_fn."""
import numpy as np
from morie.fn.teagr import teager_energy_fn, alias
from morie.fn._containers import SignalResult


class TestTeagerEnergyFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = teager_energy_fn(x)
        assert isinstance(result, SignalResult)

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = teager_energy_fn(x)
        assert result.filtered is not None

    def test_n_samples(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = teager_energy_fn(x)
        assert result.n_samples == 256

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = teager_energy_fn(x)
        assert result.name == "teager_energy"

    def test_alias(self):
        assert alias is teager_energy_fn
