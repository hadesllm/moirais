"""Tests for morie.fn.rnorm — normal random sample."""

import numpy as np
import pytest

from morie.fn.rnorm import rnorm


class TestRnorm:
    """Tests for rnorm()."""

    def test_length(self):
        """Output has correct length."""
        result = rnorm(100, seed=42)
        assert len(result) == 100

    def test_mean_approx_zero(self):
        """Large sample mean should be near 0 for standard normal."""
        result = rnorm(10000, seed=42)
        assert np.mean(result) == pytest.approx(0.0, abs=0.1)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rnorm(10, seed=42)
        assert isinstance(result, np.ndarray)

    def test_seed_reproducibility(self):
        """Same seed produces identical output."""
        a = rnorm(50, seed=123)
        b = rnorm(50, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rnorm(0)
