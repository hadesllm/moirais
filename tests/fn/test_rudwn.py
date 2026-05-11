"""Tests for rudwn (runs up and down test)."""

import numpy as np
import pytest
from morie.fn.rudwn import rudwn


class TestRudwn:
    """Runs up and down test for randomness."""

    def test_rudwn_monotonic_increasing(self):
        """Monotonic increasing should have 1 run."""
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        result = rudwn(x)
        assert result["statistic"] == 1

    def test_rudwn_monotonic_decreasing(self):
        """Monotonic decreasing should have 1 run."""
        x = np.array([7, 6, 5, 4, 3, 2, 1])
        result = rudwn(x)
        assert result["statistic"] == 1

    def test_rudwn_alternating_up_down(self):
        """Alternating up-down should have many runs."""
        x = np.array([1, 10, 2, 9, 3, 8, 4, 7, 5])
        result = rudwn(x)
        assert result["statistic"] > 4

    def test_rudwn_expected_variance_positive(self):
        """Expected runs and variance should be positive."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = rudwn(x)
        assert result["expected_runs"] > 0
        assert result["variance"] > 0

    def test_rudwn_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        result = rudwn(x)
        required_keys = {
            "statistic",
            "expected_runs",
            "variance",
            "z_stat",
            "p_value",
            "interpretation",
        }
        assert set(result.keys()) == required_keys

    def test_rudwn_small_sample_error(self):
        """Sample size < 3 should raise error."""
        with pytest.raises(ValueError):
            rudwn(np.array([1, 2]))
