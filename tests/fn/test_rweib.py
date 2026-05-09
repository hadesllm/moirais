"""Tests for moirais.fn.rweib — Weibull random sample."""

import numpy as np
import pytest

from moirais.fn.rweib import rweib


class TestRweib:
    """Tests for rweib()."""

    def test_length(self):
        """Output has correct length."""
        result = rweib(100, shape=2, scale=1, seed=42)
        assert len(result) == 100

    def test_all_positive(self):
        """All values > 0 (Weibull support is (0, inf))."""
        result = rweib(500, shape=2, scale=1, seed=42)
        assert np.all(result > 0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rweib(10, shape=1, scale=1, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rweib(0, shape=1, scale=1)
