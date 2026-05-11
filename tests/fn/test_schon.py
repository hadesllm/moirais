"""Tests for morie.fn.schon — Schoenfeld residuals."""

import numpy as np
import pytest

from morie.fn.schon import schon


class TestSchon:
    """Tests for schon()."""

    def test_residual_dimensions(self):
        """Residuals have shape (n_events, p)."""
        rng = np.random.default_rng(42)
        n = 100
        time = rng.exponential(5, size=n)
        event = rng.binomial(1, 0.7, size=n).astype(float)
        X = rng.standard_normal((n, 3))
        beta = np.array([0.5, -0.3, 0.1])
        result = schon(time, event, X, beta)
        n_events = int(event.sum())
        assert result["residuals"].shape == (n_events, 3)
        assert len(result["event_times"]) == n_events

    def test_ph_pvalues_exist(self):
        """P-values returned for each covariate."""
        rng = np.random.default_rng(99)
        n = 80
        time = rng.exponential(3, size=n)
        event = rng.binomial(1, 0.6, size=n).astype(float)
        X = rng.standard_normal((n, 2))
        beta = np.array([0.2, -0.1])
        result = schon(time, event, X, beta)
        assert len(result["p_values"]) == 2
        assert all(0 <= v <= 1 for v in result["p_values"].values())

    def test_raises_shape_mismatch(self):
        """Mismatched beta length raises ValueError."""
        with pytest.raises(ValueError, match="beta length"):
            schon(np.array([1, 2, 3]), np.array([1, 0, 1]),
                  np.array([[1, 2], [3, 4], [5, 6]]),
                  np.array([0.1]))
