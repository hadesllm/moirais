"""Tests for morie.fn.bnd -- Manski partial identification bounds."""

import numpy as np
import pytest
from morie.fn.bnd import manski_bounds


class TestManskiBounds:
    def test_no_missing_tight_bounds(self):
        """With no missing data, bounds should equal the point estimate."""
        outcome = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        missing = np.zeros(8)
        result = manski_bounds(outcome, treatment, missing)
        # With no missing data, bounds should be tight
        assert result["width"] < 0.01

    def test_missing_widens_bounds(self):
        """Missing data should widen the bounds."""
        outcome = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        missing_none = np.zeros(8)
        missing_some = np.array([0, 0, 1, 0, 0, 1, 0, 0])
        r0 = manski_bounds(outcome, treatment, missing_none)
        r1 = manski_bounds(outcome, treatment, missing_some)
        assert r1["width"] >= r0["width"] - 1e-10

    def test_lower_leq_upper(self):
        rng = np.random.default_rng(42)
        n = 50
        result = manski_bounds(
            rng.normal(0, 1, n),
            rng.binomial(1, 0.5, n),
            rng.binomial(1, 0.2, n),
        )
        assert result["lower_bound"] <= result["upper_bound"]

    def test_all_missing_raises(self):
        with pytest.raises(ValueError, match="missing"):
            manski_bounds([1, 2], [0, 1], [1, 1])
