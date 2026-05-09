"""Tests for moirais.fn.rlnrm — lognormal random sample."""

import numpy as np
import pytest

from moirais.fn.rlnrm import rlnrm


class TestRlnrm:
    """Tests for rlnrm()."""

    def test_length(self):
        """Output has correct length."""
        result = rlnrm(100, seed=42)
        assert len(result) == 100

    def test_all_positive(self):
        """All values > 0 (lognormal support is (0, inf))."""
        result = rlnrm(500, seed=42)
        assert np.all(result > 0)

    def test_returns_ndarray(self):
        """Should return ndarray."""
        result = rlnrm(10, seed=42)
        assert isinstance(result, np.ndarray)

    def test_raises_nonpositive_n(self):
        """Should reject n <= 0."""
        with pytest.raises(ValueError):
            rlnrm(0)
