"""Tests for moirais.fn.ssiz — Sample size for means."""

import pytest

from moirais.fn.ssiz import sample_size_means


class TestSampleSizeMeans:
    def test_known_case(self):
        res = sample_size_means(0.5, 1.0, alpha=0.05, power=0.80)
        assert res.estimate > 50

    def test_larger_effect_needs_fewer(self):
        r1 = sample_size_means(0.5, 1.0)
        r2 = sample_size_means(1.0, 1.0)
        assert r1.estimate > r2.estimate

    def test_invalid(self):
        with pytest.raises(ValueError):
            sample_size_means(0, 1.0)
