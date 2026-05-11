"""Tests for morie.fn.dunf — uniform PDF."""

import numpy as np
import pytest

from morie.fn.dunf import dunif


class TestDunif:
    """Tests for dunif()."""

    def test_standard_inside(self):
        """dunif(0.5) = 1.0 for standard uniform [0,1]."""
        assert dunif(0.5) == pytest.approx(1.0, abs=1e-12)

    def test_standard_at_boundary(self):
        """dunif(0.0) = 1.0 for standard uniform."""
        assert dunif(0.0) == pytest.approx(1.0, abs=1e-12)

    def test_outside_support(self):
        """dunif(-0.5) = 0.0 (outside support)."""
        assert dunif(-0.5) == pytest.approx(0.0, abs=1e-12)

    def test_raises_min_ge_max(self):
        """Should reject min >= max."""
        with pytest.raises(ValueError):
            dunif(0.5, min=1.0, max=0.0)
