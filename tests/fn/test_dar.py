"""Tests for moirais.fn.dar -- direct age-adjustment."""

import numpy as np
import pytest
from moirais.fn.dar import direct_age_adjustment


class TestDAR:
    def test_simple(self):
        """2-group example."""
        rates = np.array([0.01, 0.05])
        pop_w = np.array([5000, 3000])
        std_w = np.array([6000, 4000])
        res = direct_age_adjustment(rates, pop_w, std_w)
        assert res.measure == "DAR"
        w = np.array([0.6, 0.4])
        expected = (0.6 * 0.01 + 0.4 * 0.05) * 100_000
        assert res.estimate == pytest.approx(expected, rel=0.01)

    def test_ci(self):
        """CI should bracket estimate."""
        rates = np.array([0.01, 0.05])
        pop_w = np.array([5000, 3000])
        std_w = np.array([6000, 4000])
        res = direct_age_adjustment(rates, pop_w, std_w)
        assert res.ci_lower < res.estimate
        assert res.ci_upper > res.estimate

    def test_length_mismatch_raises(self):
        """Unequal arrays should raise."""
        with pytest.raises(ValueError):
            direct_age_adjustment(np.array([0.01]), np.array([5000, 3000]), np.array([6000]))
