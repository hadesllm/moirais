"""Tests for moirais.fn.dexp — exponential PDF."""

import numpy as np
import pytest

from moirais.fn.dexp import dexp


class TestDexp:
    """Tests for dexp()."""

    def test_at_zero(self):
        """dexp(0, rate=1) == 1.0 (PDF at origin for rate=1)."""
        assert dexp(0, rate=1.0) == pytest.approx(1.0, abs=1e-12)

    def test_at_zero_rate2(self):
        """dexp(0, rate=2) == 2.0."""
        assert dexp(0, rate=2.0) == pytest.approx(2.0, abs=1e-12)

    def test_decays(self):
        """PDF should decrease for x > 0."""
        vals = [dexp(x, rate=1.0) for x in [0, 1, 2, 3]]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_nonnegative_density(self):
        """PDF is non-negative for all x."""
        x = np.linspace(-1, 10, 100)
        result = dexp(x, rate=0.5)
        assert np.all(result >= 0)

    def test_array_input(self):
        """Should handle array input."""
        x = np.array([0, 1, 2])
        result = dexp(x, rate=1.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_raises_on_nonpositive_rate(self):
        """Should reject rate <= 0."""
        with pytest.raises(ValueError):
            dexp(1, rate=0)
        with pytest.raises(ValueError):
            dexp(1, rate=-1)
