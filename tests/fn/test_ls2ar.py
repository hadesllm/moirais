"""Test lsf_to_ar (ls2ar)."""
import numpy as np
from morie.fn.ls2ar import lsf_to_ar, ls2ar
from morie.fn._containers import DescriptiveResult


class TestLs2ar:
    def test_basic(self):
        lsf = [0.5, 1.2]
        result = lsf_to_ar(lsf)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "lsf_to_ar"
        ar = result.extra["ar"]
        assert abs(ar[0] - 1.0) < 1e-10

    def test_alias(self):
        assert ls2ar is lsf_to_ar
