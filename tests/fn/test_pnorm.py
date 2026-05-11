"""Tests for morie.fn.pnorm — normal CDF."""

import numpy as np
import pytest

from morie.fn.pnorm import pnorm


class TestPnorm:
    """Tests for pnorm()."""

    def test_at_zero(self):
        """pnorm(0) = 0.5 for standard normal."""
        assert pnorm(0) == pytest.approx(0.5, abs=1e-12)

    def test_at_196(self):
        """pnorm(1.96) ~ 0.975."""
        assert pnorm(1.96) == pytest.approx(0.975, abs=1e-3)

    def test_upper_tail(self):
        """pnorm(0, lower_tail=False) = 0.5."""
        assert pnorm(0, lower_tail=False) == pytest.approx(0.5, abs=1e-12)

    def test_type(self):
        """Scalar input returns float."""
        result = pnorm(0.0)
        assert isinstance(result, (float, np.floating))

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pnorm(x) for x in [-2, -1, 0, 1, 2]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_nonpositive_sd(self):
        """Should reject sd <= 0."""
        with pytest.raises(ValueError):
            pnorm(0, sd=0)
