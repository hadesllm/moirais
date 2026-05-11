"""Tests for morie.fn.bomg — Bayesian McDonald's omega."""

import numpy as np
import pytest
from morie.fn.bomg import bayesian_omega


class TestBayesianOmega:

    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_omega(mapq_df[items], n_iter=200)
        assert "mean" in result and "posterior" in result

    def test_mean_positive(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_omega(mapq_df[items], n_iter=200)
        if np.isfinite(result["mean"]):
            assert result["mean"] > 0

    def test_ci_ordered(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_omega(mapq_df[items], n_iter=200)
        if np.isfinite(result["ci_lower"]):
            assert result["ci_lower"] <= result["ci_upper"]

    def test_single_item_nan(self, rng):
        data = rng.standard_normal((50, 1))
        result = bayesian_omega(data, n_iter=100)
        assert np.isnan(result["mean"])

    def test_n_factors_stored(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_omega(mapq_df[items], n_iter=100, n_factors=2)
        assert result["n_factors"] == 2
