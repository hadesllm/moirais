"""Test bandwidth_compute (bndw)."""
import numpy as np
from moirais.fn.bndw import bandwidth_compute, bndw
from moirais.fn._containers import DescriptiveResult


class TestBndw:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = bandwidth_compute(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bandwidth_compute"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = bandwidth_compute(x)
        assert result.value >= 0

    def test_alias(self):
        assert bndw is bandwidth_compute
