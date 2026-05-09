"""Test crest_factor (scrst)."""
import numpy as np
from moirais.fn.scrst import crest_factor, scrst
from moirais.fn._containers import DescriptiveResult


class TestCrestFactor:
    def test_dc_signal(self):
        x = np.array([3.0, 3.0, 3.0])
        result = crest_factor(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_positive(self):
        x = np.array([0.0, 0.0, 5.0])
        assert crest_factor(x).value > 1.0

    def test_alias(self):
        assert scrst is crest_factor
