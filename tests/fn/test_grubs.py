"""Tests for morie.fn.grubs -- Grubbs' test."""

import numpy as np
import pytest
from morie.fn.grubs import grubbs_test
from morie.fn._containers import TestResult


class TestGrubbs:
    def test_clear_outlier(self):
        """Data with an obvious outlier should be detected."""
        x = [1, 2, 3, 2, 1, 3, 2, 100]
        r = grubbs_test(x)
        assert isinstance(r, TestResult)
        assert r.extra["outlier_value"] == 100.0
        assert r.p_value < 0.05

    def test_no_outlier(self):
        """Uniform-ish data should not flag outlier."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        r = grubbs_test(x)
        assert r.p_value > 0.05

    def test_zero_variance(self):
        """Constant data => p=1."""
        r = grubbs_test([5, 5, 5, 5])
        assert r.statistic == 0.0
        assert r.p_value == 1.0

    def test_raises_small_n(self):
        with pytest.raises(ValueError):
            grubbs_test([1, 2])
