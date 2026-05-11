"""Tests for morie.fn.rubin -- Rubin's rules for pooling MI estimates."""

import math

import numpy as np
import pytest
from morie.fn.rubin import rubins_rules


class TestRubinsRules:
    def test_pooled_estimate_is_mean(self):
        """Pooled estimate should equal the mean of the input estimates."""
        result = rubins_rules([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
        assert result["pooled_estimate"] == pytest.approx(2.0)

    def test_ci_contains_estimate(self):
        """CI should bracket the pooled estimate."""
        result = rubins_rules([1.0, 1.1, 0.9, 1.05, 0.95], [0.2, 0.2, 0.2, 0.2, 0.2])
        assert result["ci_lower"] < result["pooled_estimate"] < result["ci_upper"]

    def test_identical_estimates_narrow_ci(self):
        """When all estimates agree, between-variance is 0 and CI is narrow."""
        result = rubins_rules([5.0, 5.0, 5.0, 5.0, 5.0], [0.1, 0.1, 0.1, 0.1, 0.1])
        assert result["between_var"] == pytest.approx(0.0)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 1.0

    def test_too_few_raises(self):
        """Fewer than 2 estimates should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            rubins_rules([1.0], [0.5])
