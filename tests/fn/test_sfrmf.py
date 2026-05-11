"""Test form_factor (sfrmf)."""
import numpy as np
from morie.fn.sfrmf import form_factor, sfrmf
from morie.fn._containers import DescriptiveResult


class TestFormFactor:
    def test_dc_signal(self):
        x = np.array([3.0, 3.0, 3.0])
        result = form_factor(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_ge_one(self):
        x = np.array([1.0, -1.0, 2.0, -2.0])
        assert form_factor(x).value >= 1.0

    def test_alias(self):
        assert sfrmf is form_factor
