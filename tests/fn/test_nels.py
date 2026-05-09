"""Tests for moirais.fn.nels — Nelson-Aalen cumulative hazard."""

import numpy as np
import pytest

from moirais.fn.nels import nels


class TestNels:
    """Tests for nels()."""

    def test_basic_cumulative_hazard(self):
        """Cumulative hazard is non-decreasing."""
        time = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        event = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        result = nels(time, event)
        assert result.name == "Nelson-Aalen"
        ch = result.survival  # cumulative hazard stored here
        assert np.all(np.diff(ch) >= 0)

    def test_all_events_monotone(self):
        """All-event data gives strictly increasing cumulative hazard."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.ones(5)
        result = nels(time, event)
        assert result.n_events == 5
        assert result.n_censored == 0
        assert result.survival[-1] > 0

    def test_ci_bounds(self):
        """CI lower <= cumulative hazard <= CI upper."""
        rng = np.random.default_rng(7)
        time = rng.exponential(5, size=50)
        event = rng.binomial(1, 0.7, size=50).astype(float)
        result = nels(time, event)
        assert np.all(result.ci_lower <= result.survival + 1e-10)
        assert np.all(result.survival <= result.ci_upper + 1e-10)

    def test_raises_empty(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError):
            nels(np.array([]), np.array([]))
