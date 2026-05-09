"""Tests for moirais.fn.phi -- Phi coefficient for 2x2 tables."""

import pytest
from moirais.fn.phi import phi_coefficient


class TestPhiCoefficient:
    def test_perfect_positive(self):
        """Diagonal table gives phi = 1."""
        table = [[50, 0], [0, 50]]
        phi = phi_coefficient(table)
        assert isinstance(phi, float)
        assert phi == pytest.approx(1.0, abs=1e-10)

    def test_independent_zero(self):
        """Proportional table gives phi = 0."""
        table = [[25, 25], [25, 25]]
        phi = phi_coefficient(table)
        assert phi == pytest.approx(0.0, abs=1e-10)

    def test_raises_on_wrong_shape(self):
        """Non-2x2 table should raise ValueError."""
        with pytest.raises(ValueError):
            phi_coefficient([[1, 2, 3], [4, 5, 6]])
