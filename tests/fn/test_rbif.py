"""Tests for rbif -- bifactor omega."""
import numpy as np
from morie.fn.rbif import bifactor_omega
from morie.fn._containers import ESRes


class TestBifactorOmega:
    def test_basic(self):
        lg = np.array([0.5, 0.6, 0.4])
        ls = np.array([0.4, 0.3, 0.5])
        ev = np.array([0.1, 0.1, 0.1])
        result = bifactor_omega(lg, ls, ev)
        assert isinstance(result, ESRes)
        assert result.estimate > 0

    def test_omega_h(self):
        lg = np.array([0.7, 0.7])
        ls = np.array([0.3, 0.3])
        ev = np.array([0.1, 0.1])
        result = bifactor_omega(lg, ls, ev)
        assert "omega_h" in result.extra
        assert result.extra["omega_h"] > 0
