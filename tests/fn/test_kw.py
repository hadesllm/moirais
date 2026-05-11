"""Tests for morie.fn.kw -- Kruskal-Wallis H-test."""

import numpy as np
import pytest
from morie.fn.kw import kruskal_wallis_test


class TestKruskalWallis:
    def test_different_groups(self, rng):
        """Well-separated groups should give small p-value."""
        a = rng.normal(10, 1, 30)
        b = rng.normal(15, 1, 30)
        c = rng.normal(20, 1, 30)
        result = kruskal_wallis_test(a, b, c)
        assert result["p_value"] < 0.05
        assert result["df"] == 2

    def test_identical_groups(self):
        """Identical groups should give large p-value."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = kruskal_wallis_test(x, x, x)
        assert result["p_value"] > 0.05

    def test_fewer_than_2_groups_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            kruskal_wallis_test([1, 2, 3])
