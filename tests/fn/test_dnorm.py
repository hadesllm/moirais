"""Tests for morie.fn.dnorm — normal PDF."""

import numpy as np
import pytest

from morie.fn.dnorm import dnorm


class TestDnorm:
    """Tests for dnorm()."""

    def test_standard_at_zero(self):
        """dnorm(0) = 1/sqrt(2*pi) ~ 0.3989."""
        assert dnorm(0) == pytest.approx(0.3989, abs=1e-3)

    def test_nonstandard(self):
        """dnorm(0, mean=1, sd=2) ~ 0.1760."""
        assert dnorm(0, mean=1, sd=2) == pytest.approx(0.1760, abs=1e-3)

    def test_type_scalar(self):
        """Scalar input returns float."""
        result = dnorm(0.0)
        assert isinstance(result, (float, np.floating))

    def test_array_input(self):
        """Array input returns ndarray."""
        result = dnorm(np.array([-1, 0, 1]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_symmetry(self):
        """Standard normal is symmetric: dnorm(-x) == dnorm(x)."""
        assert dnorm(-1.5) == pytest.approx(dnorm(1.5), abs=1e-12)

    def test_raises_nonpositive_sd(self):
        """Should reject sd <= 0."""
        with pytest.raises(ValueError):
            dnorm(0, sd=0)
        with pytest.raises(ValueError):
            dnorm(0, sd=-1)
