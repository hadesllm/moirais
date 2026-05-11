"""Tests for morie.fn.pgam — gamma CDF."""

import numpy as np
import pytest

from morie.fn.pgam import pgamma


class TestPgamma:
    """Tests for pgamma()."""

    def test_exponential(self):
        """pgamma(1, shape=1, rate=1) = 1 - e^{-1} ~ 0.6321."""
        assert pgamma(1, shape=1, rate=1) == pytest.approx(0.6321, abs=1e-3)

    def test_at_zero(self):
        """pgamma(0, shape=1, rate=1) = 0.0."""
        assert pgamma(0, shape=1, rate=1) == pytest.approx(0.0, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pgamma(x, shape=2, rate=1) for x in [0, 1, 2, 5, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_nonpositive_shape(self):
        """Should reject shape <= 0."""
        with pytest.raises(ValueError):
            pgamma(1, shape=0)
