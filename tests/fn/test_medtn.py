"""Tests for moirais.fn.medtn -- Causal mediation analysis."""

import numpy as np
import pytest
from moirais.fn.medtn import causal_mediation


class TestCausalMediation:
    def test_total_equals_direct_plus_indirect(self):
        """total_effect ~ direct_effect + indirect_effect."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.5, n)
        Y = 0.3 * X + 0.4 * M + rng.normal(0, 0.5, n)
        result = causal_mediation(X, M, Y)
        total = result["total_effect"]
        direct_plus_indirect = result["direct_effect"] + result["indirect_effect"]
        assert abs(total - direct_plus_indirect) < 0.05

    def test_no_mediation_zero_indirect(self):
        """When M is independent of X, indirect effect ~ 0."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, n)
        M = rng.normal(0, 1, n)  # independent of X
        Y = 1.0 * X + rng.normal(0, 0.5, n)
        result = causal_mediation(X, M, Y)
        assert abs(result["indirect_effect"]) < 0.15

    def test_sobel_p_value_valid(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, n)
        M = 0.6 * X + rng.normal(0, 1, n)
        Y = 0.4 * M + rng.normal(0, 1, n)
        result = causal_mediation(X, M, Y)
        assert 0 <= result["sobel_p"] <= 1

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="4"):
            causal_mediation([1, 2], [3, 4], [5, 6])
