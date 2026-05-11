"""Tests for morie.fn.ssip — Sample size for proportions."""

import pytest

from morie.fn.ssip import sample_size_proportions


class TestSampleSizeProportions:
    def test_basic(self):
        res = sample_size_proportions(0.3, 0.5)
        assert res.estimate > 50

    def test_smaller_diff_needs_more(self):
        r1 = sample_size_proportions(0.3, 0.5)
        r2 = sample_size_proportions(0.3, 0.35)
        assert r2.estimate > r1.estimate

    def test_equal_proportions(self):
        with pytest.raises(ValueError):
            sample_size_proportions(0.5, 0.5)
