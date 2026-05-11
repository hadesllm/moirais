"""Tests for morie.fn.vpred — Predictive validity."""

import numpy as np
import pytest
from morie.fn.vpred import validity_predictive


class TestValidityPredictive:

    def test_logistic_auc(self, rng):
        scores = rng.standard_normal(200)
        outcome = (scores + rng.standard_normal(200) * 0.5 > 0).astype(float)
        result = validity_predictive(scores, outcome, method="logistic")
        assert "auc" in result
        assert 0.5 < result["auc"] <= 1.0

    def test_linear_r_squared(self, rng):
        scores = rng.standard_normal(100)
        outcome = 2 * scores + rng.standard_normal(100) * 0.3
        result = validity_predictive(scores, outcome, method="linear")
        assert "r_squared" in result
        assert result["r_squared"] > 0.5

    def test_returns_n(self, rng):
        scores = rng.standard_normal(50)
        outcome = rng.binomial(1, 0.5, 50).astype(float)
        result = validity_predictive(scores, outcome)
        assert result["n"] == 50

    def test_random_auc_near_half(self, rng):
        scores = rng.standard_normal(500)
        outcome = rng.binomial(1, 0.5, 500).astype(float)
        result = validity_predictive(scores, outcome)
        assert 0.3 < result["auc"] < 0.7
