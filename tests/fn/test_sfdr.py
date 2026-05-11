"""Test sfdr_compute (sfdr)."""
import numpy as np
from morie.fn.sfdr import sfdr_compute, sfdr
from morie.fn._containers import DescriptiveResult


class TestSfdr:
    def test_basic(self):
        t = np.arange(1024) / 1024.0
        x = np.sin(2 * np.pi * 50 * t)
        result = sfdr_compute(x, fs=1024.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "sfdr"
        assert result.value > 0

    def test_alias(self):
        assert sfdr is sfdr_compute
