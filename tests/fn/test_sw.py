"""Tests for morie.fn.sw -- Shapiro-Wilk test for normality."""

import numpy as np
import pytest
from morie.fn.sw import shapiro_wilk_test


class TestShapiroWilk:
    def test_normal_data(self, rng):
        """Normal data should not reject (p > 0.05)."""
        x = rng.standard_normal(100)
        result = shapiro_wilk_test(x)
        assert result["p_value"] > 0.05
        assert result["is_normal"] is True

    def test_uniform_data(self, rng):
        """Uniform data should be detected as non-normal for large n."""
        x = rng.uniform(0, 1, 200)
        result = shapiro_wilk_test(x)
        assert result["p_value"] < 0.05
        assert result["is_normal"] is False

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            shapiro_wilk_test([1.0, 2.0])
