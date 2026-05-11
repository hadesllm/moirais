"""Test wiener_hopf_solve (wnhpf)."""
import numpy as np
import pytest

from morie.fn.wnhpf import wiener_hopf_solve, wnhpf
from morie.fn._containers import DescriptiveResult


class TestWienerHopfSolve:
    def test_basic(self):
        Rxx = np.eye(4)
        rxd = np.array([1.0, 0.5, 0.25, 0.125])
        result = wiener_hopf_solve(Rxx, rxd)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "wiener_hopf"

    def test_identity_solution(self):
        Rxx = np.eye(3)
        rxd = np.array([1.0, 2.0, 3.0])
        result = wiener_hopf_solve(Rxx, rxd)
        coeffs = result.extra["coefficients"]
        np.testing.assert_allclose(coeffs, rxd)

    def test_value_is_length(self):
        Rxx = np.eye(5)
        rxd = np.ones(5)
        result = wiener_hopf_solve(Rxx, rxd)
        assert result.value == 5

    def test_alias(self):
        assert wnhpf is wiener_hopf_solve
