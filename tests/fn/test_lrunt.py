"""Tests for lrunt (longest run test)."""

import numpy as np
import pytest
from moirais.fn.lrunt import lrunt


class TestLrunt:
    """Longest run test for randomness."""

    def test_lrunt_constant_sequence(self):
        """Constant sequence should have longest run = n."""
        x = np.array([5, 5, 5, 5, 5, 5])
        result = lrunt(x)
        assert result["statistic"] == 6

    def test_lrunt_alternating_sequence(self):
        """Alternating sequence should have longest run = 1."""
        x = np.array([1, 10, 1, 10, 1, 10])
        result = lrunt(x)
        assert result["statistic"] == 1

    def test_lrunt_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        result = lrunt(x)
        required_keys = {"statistic", "n", "n_plus", "n_minus", "critical_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_lrunt_sample_size(self):
        """n should match input length."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = lrunt(x)
        assert result["n"] == 8

    def test_lrunt_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            lrunt(np.array([1]))
