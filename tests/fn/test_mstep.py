"""Test max_step_size (mstep)."""
import numpy as np
from moirais.fn.mstep import max_step_size, mstep
from moirais.fn._containers import DescriptiveResult


class TestMstep:
    def test_basic(self):
        x = np.ones(100)
        result = max_step_size(x, order=10)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "max_step_size"
        assert abs(result.value - 0.2) < 1e-10

    def test_random(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = max_step_size(x, order=16)
        assert result.value > 0

    def test_alias(self):
        assert mstep is max_step_size
