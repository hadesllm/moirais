"""Tests for morie.fn.pweib — Weibull CDF."""

import numpy as np
import pytest

from morie.fn.pweib import pweib


class TestPweib:
    """Tests for pweib()."""

    def test_exponential_case(self):
        """pweib(1, shape=1, scale=1) = 1 - e^{-1} ~ 0.6321."""
        assert pweib(1, shape=1, scale=1) == pytest.approx(0.6321, abs=1e-3)

    def test_at_zero(self):
        """pweib(0) = 0.0."""
        assert pweib(0, shape=1, scale=1) == pytest.approx(0.0, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [pweib(x, shape=2, scale=1) for x in [0, 0.5, 1, 2, 5]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_nonpositive_shape(self):
        """Should reject shape <= 0."""
        with pytest.raises(ValueError):
            pweib(1, shape=0)
