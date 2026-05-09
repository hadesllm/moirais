"""Tests for moirais.fn.brop — ROPE analysis."""

import numpy as np
import pytest

from moirais.fn.brop import bayesian_rope


class TestBayesianROPE:
    def test_inside_rope(self):
        samples = np.random.default_rng(42).normal(0, 0.01, 10000)
        res = bayesian_rope(samples, -0.1, 0.1)
        assert res.value > 0.95
        assert res.extra["decision"] == "accept_null"

    def test_outside_rope(self):
        samples = np.random.default_rng(42).normal(5, 0.1, 10000)
        res = bayesian_rope(samples, -0.1, 0.1)
        assert res.value < 0.05
        assert res.extra["decision"] == "reject_null"

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            bayesian_rope([1, 2, 3], 0.5, 0.1)
