"""Tests for moirais.fn.bppc — Posterior predictive check."""

import numpy as np
import pytest
from moirais.fn.bppc import bayesian_ppc


class TestBayesianPpc:

    def test_returns_dict(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        model_fit = {
            "loadings": {"EE": {f"EE{i}": 0.7 for i in range(1, 6)}},
            "residual_var": {f"EE{i}": 0.5 for i in range(1, 6)},
        }
        result = bayesian_ppc(mapq_df[items], model_fit, n_rep=50)
        assert "ppp_mean" in result

    def test_ppp_in_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        model_fit = {
            "loadings": {"EE": {f"EE{i}": 0.7 for i in range(1, 6)}},
            "residual_var": {f"EE{i}": 0.5 for i in range(1, 6)},
        }
        result = bayesian_ppc(mapq_df[items], model_fit, n_rep=50)
        assert 0 <= result["ppp_mean"] <= 1
        assert 0 <= result["ppp_var"] <= 1

    def test_observed_stats(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        model_fit = {
            "loadings": {"EE": {f"EE{i}": 0.7 for i in range(1, 6)}},
            "residual_var": {f"EE{i}": 0.5 for i in range(1, 6)},
        }
        result = bayesian_ppc(mapq_df[items], model_fit, n_rep=20)
        assert "means" in result["observed_stats"]
        assert len(result["observed_stats"]["means"]) == 5
