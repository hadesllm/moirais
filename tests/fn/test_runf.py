"""Tests for morie.fn.runf — uniform random sample."""

import numpy as np
import pytest

from morie.fn.runf import runif


class TestRunif:
    """Tests for runif()."""

    def test_length(self):
        """Output has correct length."""
        result = runif(100, seed=42)
        assert len(result) == 100

    def test_range(self):
        """All values in [0, 1] for standard uniform."""
        result = runif(1000, seed=42)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = runif(10, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            runif(0)
