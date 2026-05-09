"""Tests for moirais.fn.bfsc — Bayesian factor scores."""

import numpy as np
import pytest
from moirais.fn.bfsc import bayesian_factor_scores


class TestBayesianFactorScores:

    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        loadings = np.array([[0.7], [0.6], [0.8], [0.5], [0.7]])
        result = bayesian_factor_scores(mapq_df[items], loadings, n_iter=100)
        assert "scores" in result and "scores_sd" in result

    def test_scores_shape(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        loadings = np.array([[0.7], [0.6], [0.8], [0.5], [0.7]])
        result = bayesian_factor_scores(mapq_df[items], loadings, n_iter=100)
        assert result["scores"].shape == (200, 1)
        assert result["scores_sd"].shape == (200, 1)

    def test_sd_positive(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        loadings = np.array([[0.7], [0.6], [0.8], [0.5], [0.7]])
        result = bayesian_factor_scores(mapq_df[items], loadings, n_iter=100)
        assert np.all(result["scores_sd"] > 0)

    def test_dict_loadings(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        loadings = {"EE": {f"EE{i}": 0.7 for i in range(1, 6)}}
        result = bayesian_factor_scores(mapq_df[items], loadings, n_iter=100)
        assert result["n_factors"] == 1

    def test_n_k(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        loadings = np.array([[0.7], [0.6], [0.8], [0.5], [0.7]])
        result = bayesian_factor_scores(mapq_df[items], loadings, n_iter=50)
        assert result["n"] == 200
        assert result["k"] == 5
