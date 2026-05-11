"""Tests for morie.fn.punf — uniform CDF."""

import numpy as np
import pytest

from morie.fn.punf import punif


class TestPunif:
    """Tests for punif()."""

    def test_midpoint(self):
        """punif(0.5) = 0.5 for standard uniform."""
        assert punif(0.5) == pytest.approx(0.5, abs=1e-12)

    def test_at_one(self):
        """punif(1.0) = 1.0."""
        assert punif(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_at_zero(self):
        """punif(0.0) = 0.0."""
        assert punif(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_raises_min_ge_max(self):
        """Should reject min >= max."""
        with pytest.raises(ValueError):
            punif(0.5, min=2.0, max=1.0)
