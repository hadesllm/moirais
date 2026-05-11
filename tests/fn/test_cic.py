"""Tests for morie.fn.cic -- Changes-in-Changes estimator."""

import numpy as np
import pytest
from morie.fn.cic import changes_in_changes


class TestCIC:
    def test_no_effect_ate_near_zero(self):
        """When treatment has no effect, ATE should be near 0."""
        rng = np.random.default_rng(42)
        n = 200
        group = np.array([0]*100 + [1]*100)
        time = np.tile([0, 1], 100)
        outcome = rng.normal(5, 1, n)
        result = changes_in_changes(outcome, group, time)
        assert abs(result["ate"]) < 1.0

    def test_positive_effect(self):
        """With a clear treatment effect, ATE should be positive."""
        rng = np.random.default_rng(42)
        n = 400
        group = np.array([0]*200 + [1]*200)
        time = np.tile([0, 1], 200)
        outcome = rng.normal(5, 1, n)
        # Add effect to treated-post
        mask = (group == 1) & (time == 1)
        outcome[mask] += 3.0
        result = changes_in_changes(outcome, group, time)
        assert result["ate"] > 1.0

    def test_quantile_effects_length(self):
        rng = np.random.default_rng(42)
        n = 200
        group = np.array([0]*100 + [1]*100)
        time = np.tile([0, 1], 100)
        outcome = rng.normal(0, 1, n)
        result = changes_in_changes(outcome, group, time, n_quantiles=50)
        assert len(result["quantile_effects"]) == 50
