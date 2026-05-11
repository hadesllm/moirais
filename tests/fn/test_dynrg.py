"""Test dynamic_range (dynrg)."""
import numpy as np
from morie.fn.dynrg import dynamic_range, dynrg
from morie.fn._containers import DescriptiveResult


class TestDynrg:
    def test_basic(self):
        x = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        result = dynamic_range(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "dynamic_range"
        assert result.value > 0

    def test_known_value(self):
        x = np.array([1.0, 100.0])
        result = dynamic_range(x)
        assert abs(result.value - 40.0) < 1e-10

    def test_alias(self):
        assert dynrg is dynamic_range
