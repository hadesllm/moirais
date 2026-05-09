"""Tests for moirais.fn.plnrm — lognormal CDF."""

import numpy as np
import pytest

from moirais.fn.plnrm import plnrm


class TestPlnrm:
    """Tests for plnrm()."""

    def test_at_one(self):
        """plnrm(1, meanlog=0, sdlog=1) = 0.5 (median of lognormal)."""
        assert plnrm(1, meanlog=0, sdlog=1) == pytest.approx(0.5, abs=1e-12)

    def test_at_zero(self):
        """plnrm(0) = 0.0."""
        assert plnrm(0, meanlog=0, sdlog=1) == pytest.approx(0.0, abs=1e-12)

    def test_monotone(self):
        """CDF is non-decreasing."""
        vals = [plnrm(x, meanlog=0, sdlog=1) for x in [0.1, 0.5, 1, 2, 10]]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_raises_nonpositive_sdlog(self):
        """Should reject sdlog <= 0."""
        with pytest.raises(ValueError):
            plnrm(1, sdlog=0)
