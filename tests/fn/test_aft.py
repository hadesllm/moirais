"""Tests for moirais.fn.aft — Accelerated Failure Time model."""

import numpy as np
import pytest

from moirais.fn.aft import aft


class TestAft:
    """Tests for aft()."""

    def test_basic_weibull(self):
        """Fits on synthetic Weibull data without error."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 2))
        log_t = 2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * rng.gumbel(size=n)
        time = np.exp(log_t)
        event = rng.binomial(1, 0.8, size=n).astype(float)
        result = aft(time, event, X)
        assert "coefficients" in result
        assert result["n"] == n
        assert result["distribution"] == "weibull"
        assert len(result["coefficients"]) == 3  # intercept + 2

    def test_coefficients_direction(self):
        """Positive covariate effect shows positive coefficient."""
        rng = np.random.default_rng(123)
        n = 300
        x = rng.standard_normal(n)
        log_t = 1.0 + 1.0 * x + 0.3 * rng.gumbel(size=n)
        time = np.exp(log_t)
        event = np.ones(n)
        result = aft(time, event, x.reshape(-1, 1))
        assert result["coefficients"]["x0"] > 0.3  # should recover ~1.0

    def test_raises_nonpositive_time(self):
        """Rejects non-positive times."""
        with pytest.raises(ValueError, match="times must be"):
            aft(np.array([0, 1, 2]), np.array([1, 1, 0]),
                np.array([[1], [2], [3]]))

    def test_raises_unsupported_distribution(self):
        """Rejects unknown distribution."""
        with pytest.raises(ValueError, match="Unsupported"):
            aft(np.array([1.0, 2.0]), np.array([1, 1]),
                np.array([[1], [2]]), distribution="exponential")
