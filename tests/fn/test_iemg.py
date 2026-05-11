"""Test integrated_emg_fn."""
import numpy as np
from morie.fn.iemg import integrated_emg_fn, alias
from morie.fn._containers import DescriptiveResult


class TestIntegratedEmgFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = integrated_emg_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = integrated_emg_fn(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_known_value(self):
        x = np.ones(256)
        result = integrated_emg_fn(x)
        assert abs(result.value - 256.0) < 1e-9

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = integrated_emg_fn(x)
        assert result.name == "integrated_emg"

    def test_alias(self):
        assert alias is integrated_emg_fn
