"""Tests for moirais.fn.rbnm — binomial random sample."""

import numpy as np
import pytest

from moirais.fn.rbnm import rbinom


class TestRbinom:
    """Tests for rbinom()."""

    def test_length(self):
        """Output has correct length."""
        result = rbinom(100, size=10, prob=0.5, seed=42)
        assert len(result) == 100

    def test_range(self):
        """All values in [0, size]."""
        result = rbinom(200, size=10, prob=0.5, seed=42)
        assert np.all(result >= 0)
        assert np.all(result <= 10)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rbinom(10, size=5, prob=0.3, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rbinom(0, size=10, prob=0.5)
