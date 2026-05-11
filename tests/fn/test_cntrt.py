"""Test centroidal_time."""
import numpy as np
from morie.fn.cntrt import centroidal_time, cntrt
from morie.fn._containers import DescriptiveResult


class TestCentroidalTime:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = centroidal_time(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_nonneg(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = centroidal_time(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_impulse_at_start(self):
        x = np.zeros(100)
        x[0] = 10.0
        result = centroidal_time(x, fs=1.0)
        assert result.value < 1.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = centroidal_time(x)
        assert result.name == "centroidal_time"

    def test_alias(self):
        assert cntrt is centroidal_time
