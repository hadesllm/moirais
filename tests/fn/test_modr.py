"""Tests for morie.fn.modr — moderation analysis."""
import numpy as np
import pytest
from morie.fn.modr import moderation_analysis


class TestModerationAnalysis:
    def test_interaction(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        mod = rng.standard_normal(n)
        y = 0.5 * x + 0.3 * mod + 0.8 * x * mod + rng.standard_normal(n) * 0.5
        res = moderation_analysis(y, x, mod)
        assert res.extra["interaction_coef"] != 0
        assert res.extra["interaction_p"] < 0.05

    def test_no_interaction(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        mod = rng.standard_normal(n)
        y = 0.5 * x + 0.3 * mod + rng.standard_normal(n) * 2.0
        res = moderation_analysis(y, x, mod)
        assert np.isfinite(res.extra["interaction_coef"])
