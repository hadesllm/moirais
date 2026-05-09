"""Tests for moirais.fn.bmi — Bayesian measurement invariance."""

import numpy as np
import pytest
from moirais.fn.bmi import bayesian_mi


class TestBayesianMi:

    def test_returns_dict(self, mapq_df):
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        group = mapq_df["gender"].values
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_mi(mapq_df[items], group, structure, n_iter=200)
        assert "configural" in result
        assert "metric_invariance" in result

    def test_metric_invariance_bool(self, mapq_df):
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        group = mapq_df["gender"].values
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_mi(mapq_df[items], group, structure, n_iter=200)
        assert isinstance(result["metric_invariance"], bool)

    def test_scalar_invariance_bool(self, mapq_df):
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        group = mapq_df["gender"].values
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_mi(mapq_df[items], group, structure, n_iter=200)
        assert isinstance(result["scalar_invariance"], bool)

    def test_n_groups(self, mapq_df):
        structure = {"EE": [f"EE{i}" for i in range(1, 6)]}
        group = mapq_df["gender"].values
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = bayesian_mi(mapq_df[items], group, structure, n_iter=100)
        assert result["n_groups"] == 2
