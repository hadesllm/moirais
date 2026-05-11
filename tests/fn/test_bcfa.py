"""Tests for morie.fn.bcfa — Bayesian CFA."""

import numpy as np
import pytest
from morie.fn.bcfa import bayesian_cfa


class TestBayesianCfa:

    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        structure = {
            "EE": [f"EE{i}" for i in range(1, 6)],
            "EA": [f"EA{i}" for i in range(1, 6)],
        }
        result = bayesian_cfa(mapq_df[items], structure, n_iter=200)
        assert "loadings" in result and "ppp" in result

    def test_loadings_per_factor(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = bayesian_cfa(mapq_df[items], structure, n_iter=200)
        assert "EE" in result["loadings"]
        assert len(result["loadings"]["EE"]) == 5

    def test_ppp_in_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = bayesian_cfa(mapq_df[items], structure, n_iter=200)
        assert 0 <= result["ppp"] <= 1

    def test_residual_var_positive(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        result = bayesian_cfa(mapq_df[items], structure, n_iter=200)
        for v in result["residual_var"].values():
            assert v > 0

    def test_n_factors(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        structure = {
            "EE": [f"EE{i}" for i in range(1, 6)],
            "EA": [f"EA{i}" for i in range(1, 6)],
        }
        result = bayesian_cfa(mapq_df[items], structure, n_iter=100)
        assert result["n_factors"] == 2
