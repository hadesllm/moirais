"""Tests for moirais.fn.cumhz — cumulative hazard from KM."""

import numpy as np
import pytest

from moirais.fn.cumhz import cumhz


class TestCumhz:
    """Tests for cumhz()."""

    def test_basic_identity(self):
        """H(t) = -log(S(t)) for known survival."""
        times = np.array([1.0, 2.0, 3.0])
        survival = np.array([0.9, 0.7, 0.5])
        result = cumhz(times, survival)
        expected = -np.log(survival)
        np.testing.assert_allclose(result["cumulative_hazard"], expected)

    def test_survival_one_gives_zero(self):
        """S(t)=1.0 gives H(t)=0."""
        result = cumhz(np.array([1.0]), np.array([1.0]))
        assert result["cumulative_hazard"][0] == pytest.approx(0.0)

    def test_raises_on_zero_survival(self):
        """S(t)=0 is invalid."""
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            cumhz(np.array([1.0]), np.array([0.0]))

    def test_raises_on_negative_survival(self):
        """Negative survival is invalid."""
        with pytest.raises(ValueError):
            cumhz(np.array([1.0]), np.array([-0.5]))
