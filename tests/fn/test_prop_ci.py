"""Tests for morie.fn.prop_ci -- proportion confidence interval."""

import pytest
from morie.fn.prop_ci import proportion_ci


class TestProportionCI:
    def test_half_proportion_wilson(self):
        """50 successes out of 100 should give CI around 0.5."""
        lo, hi = proportion_ci(50, 100)
        assert lo < 0.5 < hi
        assert lo > 0.35
        assert hi < 0.65

    def test_clopper_pearson(self):
        """Clopper-Pearson should be wider (more conservative)."""
        lo_w, hi_w = proportion_ci(50, 100, method="wilson")
        lo_cp, hi_cp = proportion_ci(50, 100, method="clopper-pearson")
        assert (hi_cp - lo_cp) >= (hi_w - lo_w) - 0.01  # CP at least as wide

    def test_zero_successes(self):
        """Zero successes should have lower bound at 0."""
        lo, hi = proportion_ci(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-10)
        assert hi > 0.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            proportion_ci(5, 0)
