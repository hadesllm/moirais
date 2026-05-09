"""Tests for moirais.fn.rgeom — geometric random sample."""

import numpy as np
import pytest

from moirais.fn.rgeom import rgeom


class TestRgeom:
    """Tests for rgeom()."""

    def test_length(self):
        """Output has correct length."""
        result = rgeom(100, prob=0.5, seed=42)
        assert len(result) == 100

    def test_nonnegative(self):
        """All values >= 0 (number of failures)."""
        result = rgeom(200, prob=0.3, seed=42)
        assert np.all(result >= 0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rgeom(10, prob=0.5, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rgeom(0, prob=0.5)
