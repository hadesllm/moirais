"""Tests for moirais.fn.bdif — Bayesian DIF detection."""

import numpy as np
import pytest
from moirais.fn.bdif import bayesian_dif


class TestBayesianDif:

    def test_returns_dict(self, mapq_binary_df, rng):
        group = rng.choice([0, 1], len(mapq_binary_df))
        result = bayesian_dif(mapq_binary_df, group, n_iter=100)
        assert "items" in result and "n_flagged" in result

    def test_items_per_item(self, mapq_binary_df, rng):
        group = rng.choice([0, 1], len(mapq_binary_df))
        result = bayesian_dif(mapq_binary_df, group, n_iter=100)
        assert len(result["items"]) == 10

    def test_each_item_has_ci(self, mapq_binary_df, rng):
        group = rng.choice([0, 1], len(mapq_binary_df))
        result = bayesian_dif(mapq_binary_df, group, n_iter=100)
        for item in result["items"]:
            assert "ci_lower" in item and "ci_upper" in item

    def test_wrong_groups_raises(self, mapq_binary_df):
        group = np.array([0] * len(mapq_binary_df))
        with pytest.raises(ValueError):
            bayesian_dif(mapq_binary_df, group, n_iter=50)

    def test_n_k(self, mapq_binary_df, rng):
        group = rng.choice([0, 1], len(mapq_binary_df))
        result = bayesian_dif(mapq_binary_df, group, n_iter=50)
        assert result["n"] == 200
        assert result["k"] == 10
