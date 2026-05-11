"""Tests for morie.fn.rnbm — negative binomial random sample."""

import numpy as np
import pytest

from morie.fn.rnbm import rnbm


class TestRnbm:
    """Tests for rnbm()."""

    def test_length(self):
        """Output has correct length."""
        result = rnbm(100, size=5, prob=0.5, seed=42)
        assert len(result) == 100

    def test_nonnegative(self):
        """All values >= 0."""
        result = rnbm(200, size=3, prob=0.4, seed=42)
        assert np.all(result >= 0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rnbm(10, size=2, prob=0.5, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rnbm(0, size=1, prob=0.5)
