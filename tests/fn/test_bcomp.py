"""Tests for morie.fn.bcomp — Bayesian model comparison."""

import numpy as np
import pytest
from morie.fn.bcomp import bayesian_model_compare


class TestBayesianModelCompare:

    def test_returns_dict(self, rng):
        ll1 = rng.standard_normal((100, 50))
        ll2 = rng.standard_normal((100, 50)) - 0.1
        result = bayesian_model_compare(
            {"loglik_samples": ll1},
            {"loglik_samples": ll2},
        )
        assert "preferred" in result

    def test_preferred_string(self, rng):
        ll1 = rng.standard_normal((100, 50))
        ll2 = rng.standard_normal((100, 50))
        result = bayesian_model_compare(
            {"loglik_samples": ll1},
            {"loglik_samples": ll2},
        )
        assert result["preferred"] in ("model1", "model2", "indeterminate")

    def test_precomputed_ic(self):
        result = bayesian_model_compare(
            {"dic": 100, "waic": 105},
            {"dic": 110, "waic": 115},
        )
        assert result["delta_dic"] == -10
        assert result["delta_waic"] == -10
        assert result["preferred"] == "model1"

    def test_empty_loglik(self):
        result = bayesian_model_compare(
            {"loglik_samples": np.array([[]])},
            {"loglik_samples": np.array([[]])},
        )
        assert result["preferred"] == "indeterminate"

    def test_bf_approx_positive(self, rng):
        ll1 = rng.standard_normal((50, 30))
        ll2 = rng.standard_normal((50, 30)) - 0.5
        result = bayesian_model_compare(
            {"loglik_samples": ll1},
            {"loglik_samples": ll2},
        )
        if result["bf_approx"] is not None:
            assert result["bf_approx"] > 0
