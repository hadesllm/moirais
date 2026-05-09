"""Tests for moirais.fn.cbnd — causal bounds (Manski)."""
import numpy as np
import pytest
from moirais.fn.cbnd import causal_bounds


class TestCausalBounds:
    def test_bounds_ordered(self):
        rng = np.random.default_rng(42)
        n = 200
        treatment = rng.choice([0, 1], size=n)
        y = treatment * 0.5 + rng.standard_normal(n) * 0.3
        res = causal_bounds(y, treatment)
        assert res.extra["ate_lower"] <= res.extra["ate_upper"]

    def test_all_treated_raises(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(50)
        treatment = np.ones(50)
        with pytest.raises(ValueError, match="both treatment groups"):
            causal_bounds(y, treatment)

    def test_midpoint_between_bounds(self):
        rng = np.random.default_rng(42)
        n = 100
        treatment = rng.choice([0, 1], size=n)
        y = rng.uniform(0, 1, size=n)
        res = causal_bounds(y, treatment)
        mid = res.extra["ate_midpoint"]
        assert res.extra["ate_lower"] <= mid <= res.extra["ate_upper"]
