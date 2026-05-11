"""Tests for morie.fn.bci -- Bayesian credible interval."""

import numpy as np
import pytest
from morie.fn.bci import bayesian_credible_interval


class TestBayesianCI:
    def test_equal_tailed_contains_median(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 5000)
        lo, hi = bayesian_credible_interval(samples, alpha=0.05, method="equal_tailed")
        median = np.median(samples)
        assert lo < median < hi

    def test_hdi_method_works(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(3.0, 1.0, 5000)
        lo, hi = bayesian_credible_interval(samples, alpha=0.05, method="hdi")
        assert lo < 3.0 < hi

    def test_narrower_interval_with_higher_alpha(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 5000)
        lo95, hi95 = bayesian_credible_interval(samples, alpha=0.05)
        lo80, hi80 = bayesian_credible_interval(samples, alpha=0.20)
        assert (hi95 - lo95) > (hi80 - lo80)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            bayesian_credible_interval([1, 2, 3, 4], method="xyz")
