"""Test rms_value (srms)."""
import numpy as np
from moirais.fn.srms import rms_value, srms
from moirais.fn._containers import DescriptiveResult


class TestRMS:
    def test_dc(self):
        x = np.array([3.0, 3.0, 3.0])
        result = rms_value(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 3.0) < 1e-10

    def test_known(self):
        x = np.array([1.0, -1.0])
        assert abs(rms_value(x).value - 1.0) < 1e-10

    def test_alias(self):
        assert srms is rms_value
