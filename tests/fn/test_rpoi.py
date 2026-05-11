"""Tests for morie.fn.rpoi — Poisson random sample."""

import numpy as np
import pytest

from morie.fn.rpoi import rpois


class TestRpois:
    """Tests for rpois()."""

    def test_length(self):
        """Output has correct length."""
        result = rpois(100, lambda_=3.0, seed=42)
        assert len(result) == 100

    def test_nonnegative(self):
        """All values >= 0."""
        result = rpois(200, lambda_=5.0, seed=42)
        assert np.all(result >= 0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rpois(10, lambda_=1.0, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rpois(0, lambda_=1.0)
