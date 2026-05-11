"""Tests for morie.fn.birt — Bayesian 2PL IRT."""

import numpy as np
import pytest
from morie.fn.birt import bayesian_irt_2pl


class TestBayesianIrt2pl:

    def test_returns_dict(self, mapq_binary_df):
        result = bayesian_irt_2pl(mapq_binary_df, n_iter=100)
        assert "a_mean" in result and "b_mean" in result

    def test_a_positive(self, mapq_binary_df):
        result = bayesian_irt_2pl(mapq_binary_df, n_iter=100)
        for v in result["a_mean"].values():
            assert v > 0  # discrimination should be positive

    def test_theta_array(self, mapq_binary_df):
        result = bayesian_irt_2pl(mapq_binary_df, n_iter=100)
        assert result["theta_mean"].shape[0] == len(mapq_binary_df)

    def test_n_k(self, mapq_binary_df):
        result = bayesian_irt_2pl(mapq_binary_df, n_iter=100)
        assert result["n"] == 200
        assert result["k"] == 10

    def test_posterior_shape(self, mapq_binary_df):
        result = bayesian_irt_2pl(mapq_binary_df, n_iter=100)
        assert result["a_posterior"].shape[1] == 10
