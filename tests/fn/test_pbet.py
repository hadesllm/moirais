"""Tests for morie.fn.pbet — beta CDF."""

import numpy as np
import pytest

from morie.fn.pbet import pbeta


class TestPbeta:
    """Tests for pbeta()."""

    def test_uniform_midpoint(self):
        """pbeta(0.5, alpha=1, beta=1) = 0.5."""
        assert pbeta(0.5, alpha=1, beta=1) == pytest.approx(0.5, abs=1e-12)

    def test_at_one(self):
        """pbeta(1.0, 2, 3) = 1.0."""
        assert pbeta(1.0, alpha=2, beta=3) == pytest.approx(1.0, abs=1e-12)

    def test_at_zero(self):
        """pbeta(0.0, 2, 3) = 0.0."""
        assert pbeta(0.0, alpha=2, beta=3) == pytest.approx(0.0, abs=1e-12)

    def test_raises_nonpositive_beta(self):
        """Should reject beta <= 0."""
        with pytest.raises(ValueError):
            pbeta(0.5, alpha=1, beta=0)
