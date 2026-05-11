"""Tests for morie.fn.tthet — Jacobi theta function."""

import pytest

from morie.fn.tthet import theta_function


class TestThetaFunction:
    def test_z_zero(self):
        r = theta_function(z=0.0, tau=1j, terms=20)
        assert r.value > 1.0

    def test_returns_float(self):
        r = theta_function()
        assert isinstance(r.value, float)

    def test_invalid(self):
        with pytest.raises(ValueError):
            theta_function(tau=1.0)
