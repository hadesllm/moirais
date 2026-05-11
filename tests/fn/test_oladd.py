"""Test overlap_add (oladd)."""
import numpy as np
from morie.fn.oladd import overlap_add, oladd
from morie.fn._containers import DescriptiveResult


class TestOladd:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(128)
        h = np.array([0.25, 0.5, 0.25])
        result = overlap_add(x, h, block_size=32)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "overlap_add"

    def test_matches_direct(self):
        x = np.random.default_rng(42).standard_normal(128)
        h = np.array([0.25, 0.5, 0.25])
        result = overlap_add(x, h, block_size=32)
        direct = np.convolve(x, h)
        assert np.allclose(result.extra["output"][:len(direct)], direct, atol=1e-10)

    def test_alias(self):
        assert oladd is overlap_add
