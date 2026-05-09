"""Tests for moirais.fn.levene -- Levene's test for equality of variances."""

import numpy as np
import pytest
from moirais.fn.levene import levene_test


class TestLevene:
    def test_equal_variances(self, rng):
        """Groups with same variance should not reject."""
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        result = levene_test(a, b)
        assert result["p_value"] > 0.05

    def test_unequal_variances(self, rng):
        """Groups with very different variances should reject."""
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 10, 100)
        result = levene_test(a, b)
        assert result["p_value"] < 0.05

    def test_df_values(self, rng):
        """Check degrees of freedom for 3 groups of 40."""
        a = rng.normal(0, 1, 40)
        b = rng.normal(0, 1, 40)
        c = rng.normal(0, 1, 40)
        result = levene_test(a, b, c)
        assert result["df_between"] == 2
        assert result["df_within"] == 117
