"""Tests for moirais.fn.vtest — Test-retest reliability."""

import numpy as np
import pytest
from moirais.fn.vtest import validity_test_retest


class TestValidityTestRetest:

    def test_high_icc_correlated(self, rng):
        s1 = rng.standard_normal(100)
        s2 = s1 + rng.standard_normal(100) * 0.1
        result = validity_test_retest(s1, s2)
        assert result["icc"] > 0.8

    def test_pearson_r_present(self, rng):
        s1 = rng.standard_normal(50)
        s2 = s1 + rng.standard_normal(50) * 0.5
        result = validity_test_retest(s1, s2)
        assert np.isfinite(result["pearson_r"])

    def test_ci_contains_icc(self, rng):
        s1 = rng.standard_normal(80)
        s2 = s1 + rng.standard_normal(80) * 0.3
        result = validity_test_retest(s1, s2)
        assert result["icc_ci_lower"] <= result["icc"] <= result["icc_ci_upper"]

    def test_mean_diff_near_zero(self, rng):
        s1 = rng.standard_normal(100)
        s2 = s1 + rng.standard_normal(100) * 0.05
        result = validity_test_retest(s1, s2)
        assert abs(result["mean_diff"]) < 0.5

    def test_small_n(self):
        result = validity_test_retest(np.array([1.0]), np.array([2.0]))
        assert np.isnan(result["icc"])
