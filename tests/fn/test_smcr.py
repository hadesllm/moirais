"""Test mean_crossing_rate (smcr)."""
import numpy as np
from moirais.fn.smcr import mean_crossing_rate, smcr
from moirais.fn._containers import DescriptiveResult


class TestMCR:
    def test_oscillation(self):
        x = np.array([1.0, 3.0, 1.0, 3.0])
        result = mean_crossing_rate(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_alias(self):
        assert smcr is mean_crossing_rate
