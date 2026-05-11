"""Tests for morie.fn.balph — Bayesian Cronbach's alpha."""

import numpy as np
import pytest
from morie.fn.balph import bayesian_alpha


class TestBayesianAlpha:

    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_alpha(mapq_df[items], n_iter=200)
        assert "mean" in result and "ci_lower" in result

    def test_mean_in_ci(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_alpha(mapq_df[items], n_iter=200)
        if np.isfinite(result["mean"]):
            assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_posterior_array(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_alpha(mapq_df[items], n_iter=200)
        assert len(result["posterior"]) > 0

    def test_single_item_nan(self, rng):
        data = rng.standard_normal((50, 1))
        result = bayesian_alpha(data, n_iter=100)
        assert np.isnan(result["mean"])

    def test_n_k_correct(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_alpha(mapq_df[items], n_iter=100)
        assert result["n"] == 200
        assert result["k"] == 5
