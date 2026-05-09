"""Test reflection_to_ar (rcfcv)."""
import numpy as np
from moirais.fn.rcfcv import reflection_to_ar, rcfcv
from moirais.fn._containers import DescriptiveResult


class TestRcfcv:
    def test_basic(self):
        rc = [0.5, -0.3]
        result = reflection_to_ar(rc)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "reflection_to_ar"
        ar = result.extra["ar"]
        assert abs(ar[0] - 1.0) < 1e-10

    def test_single(self):
        result = reflection_to_ar([0.8])
        ar = result.extra["ar"]
        assert len(ar) == 2
        assert abs(ar[1] - 0.8) < 1e-10

    def test_alias(self):
        assert rcfcv is reflection_to_ar
