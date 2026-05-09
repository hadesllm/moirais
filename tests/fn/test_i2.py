"""Tests for moirais.fn.i2 -- Higgins' I-squared heterogeneity."""

import pytest
from moirais.fn.i2 import i_squared


class TestISquared:
    def test_homogeneous_studies(self):
        """Identical effects should give I^2 near 0."""
        val = i_squared(
            estimates=[1.0, 1.0, 1.0],
            standard_errors=[0.1, 0.1, 0.1],
        )
        assert isinstance(val, float)
        assert val == pytest.approx(0.0, abs=5.0)

    def test_heterogeneous_studies(self):
        """Widely different effects should give high I^2."""
        val = i_squared(
            estimates=[0.1, 1.0, 2.0],
            standard_errors=[0.05, 0.05, 0.05],
        )
        assert val > 50.0

    def test_range_0_100(self):
        """I^2 should be between 0 and 100."""
        val = i_squared([0.5, 0.6, 0.4], [0.1, 0.15, 0.12])
        assert 0 <= val <= 100
