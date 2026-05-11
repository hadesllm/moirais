"""Tests for morie.fn.irf — Impulse Response Function."""

import numpy as np
import pytest

from morie.fn.irf import irf


class TestIrf:
    """Tests for irf()."""

    def test_identity_matrix(self):
        """Identity VAR matrix: shock persists indefinitely."""
        A = np.eye(2)
        result = irf(A, periods=5, shock_var=0)
        assert result.shape == (6, 2)
        # Shock to var 0 stays at 1.0
        np.testing.assert_allclose(result[:, 0], np.ones(6))
        # Var 1 stays at 0.0
        np.testing.assert_allclose(result[:, 1], np.zeros(6))

    def test_decay(self):
        """Stable VAR matrix gives decaying responses."""
        A = np.array([[0.5, 0.1], [0.0, 0.3]])
        result = irf(A, periods=20, shock_var=0)
        # Response should decay toward zero
        assert abs(result[-1, 0]) < abs(result[0, 0])

    def test_shock_var_1(self):
        """Shock to variable 1 only."""
        A = np.array([[0.5, 0.0], [0.0, 0.8]])
        result = irf(A, periods=5, shock_var=1)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 1.0

    def test_raises_nonsquare(self):
        """Non-square matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            irf(np.array([[1, 2, 3], [4, 5, 6]]))
