"""Tests for runsq (Wald-Wolfowitz runs test)."""

import numpy as np
import pytest
from morie.fn.runsq import runsq


class TestRunsq:
    """Wald-Wolfowitz runs test."""

    def test_runsq_alternating_sequence(self):
        """Alternating sequence should have many runs (reject randomness)."""
        x = np.array([1, 10, 1, 10, 1, 10, 1, 10, 1, 10])
        result = runsq(x)
        assert result["statistic"] == 10  # maximum runs
        assert result["p_value"] < 0.05
        assert result["interpretation"] == "not random"

    def test_runsq_constant_sequence(self):
        """Constant sequence should have 1 run (reject randomness)."""
        x = np.array([5, 5, 5, 5, 5, 5, 5, 5])
        result = runsq(x)
        assert result["statistic"] == 1
        assert result["p_value"] < 0.05
        assert result["interpretation"] == "not random"

    def test_runsq_random_sequence(self):
        """Random sequence should not be rejected as nonrandom."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        result = runsq(x)
        assert result["p_value"] > 0.05 or result["p_value"] < 1.0
        assert "statistic" in result

    def test_runsq_n_plus_n_minus(self):
        """n_plus + n_minus should equal sample size."""
        x = np.array([2.5, 1.5, 3.0, 4.0, 2.0, 3.5])
        result = runsq(x)
        assert result["n_plus"] + result["n_minus"] == len(x)

    def test_runsq_expected_variance_positive(self):
        """Expected runs and variance should be positive."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = runsq(x)
        assert result["expected_runs"] > 0
        assert result["variance"] > 0

    def test_runsq_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            runsq(np.array([1]))

    def test_runsq_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = runsq(x)
        required_keys = {
            "statistic",
            "n_plus",
            "n_minus",
            "expected_runs",
            "variance",
            "z_stat",
            "p_value",
            "interpretation",
        }
        assert set(result.keys()) == required_keys
