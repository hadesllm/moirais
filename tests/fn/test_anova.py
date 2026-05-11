"""Tests for morie.fn.anova -- One-way ANOVA F-test."""

import numpy as np
import pytest
from morie.fn.anova import anova_one_way


class TestAnovaOneWay:
    def test_different_groups_significant(self):
        """Well-separated groups should yield small p-value."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [20.0, 21.0, 22.0, 23.0, 24.0]
        c = [30.0, 31.0, 32.0, 33.0, 34.0]
        result = anova_one_way(a, b, c)
        assert isinstance(result, dict)
        assert result["F"] > 10
        assert result["p_value"] < 0.001
        assert result["df_between"] == 2
        assert result["df_within"] == 12

    def test_similar_groups_not_significant(self, rng):
        """Similar groups should give large p-value."""
        a = rng.normal(5, 1, 20).tolist()
        b = rng.normal(5, 1, 20).tolist()
        c = rng.normal(5, 1, 20).tolist()
        result = anova_one_way(a, b, c)
        assert result["p_value"] > 0.01

    def test_raises_on_single_group(self):
        """Fewer than 2 groups should raise ValueError."""
        with pytest.raises(ValueError):
            anova_one_way([1, 2, 3])
