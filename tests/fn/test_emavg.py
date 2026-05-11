"""Test exponential_ma (emavg)."""
import numpy as np
from morie.fn.emavg import exponential_ma, emavg
from morie.fn._containers import DescriptiveResult


class TestExponentialMA:
    def test_length(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = exponential_ma(x, alpha=0.5)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 5

    def test_first_value(self):
        x = np.array([10.0, 20.0, 30.0])
        result = exponential_ma(x, alpha=0.5)
        assert result.value[0] == 10.0

    def test_alias(self):
        assert emavg is exponential_ma
