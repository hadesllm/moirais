"""Tests for moirais.fn.rf — F-distribution random variates."""

import numpy as np
import pytest

from moirais.fn.rf import rf_dist, rf


class TestRf:
    """Tests for rf_dist() / rf()."""

    def test_length(self):
        """Output length matches requested n."""
        result = rf_dist(100, 5, 10)
        assert len(result) == 100

    def test_mean_near_theoretical(self):
        """For dfd>2, E[X] = dfd/(dfd-2)."""
        dfd = 20
        result = rf_dist(100_000, 5, dfd, seed=0)
        expected_mean = dfd / (dfd - 2)
        assert abs(np.mean(result) - expected_mean) < 0.05

    def test_all_nonnegative(self):
        """F variates are always >= 0."""
        result = rf_dist(10_000, 3, 4, seed=5)
        assert np.all(result >= 0)

    def test_alias(self):
        """rf should be the same function as rf_dist."""
        a = rf_dist(50, 5, 5, seed=42)
        b = rf(50, 5, 5, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_reproducible(self):
        """Same seed produces identical output."""
        a = rf_dist(50, 3, 4, seed=99)
        b = rf_dist(50, 3, 4, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_raises_on_nonpositive_df(self):
        """Should reject dfn <= 0 or dfd <= 0."""
        with pytest.raises(ValueError):
            rf_dist(10, 0, 5)
        with pytest.raises(ValueError):
            rf_dist(10, 5, -1)
