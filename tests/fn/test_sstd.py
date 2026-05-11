"""Test sample_std (sstd)."""
import numpy as np
from morie.fn.sstd import sample_std, sstd
from morie.fn._containers import DescriptiveResult


class TestSampleStd:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_std(x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - np.std(x, ddof=1)) < 1e-10

    def test_alias(self):
        assert sstd is sample_std
