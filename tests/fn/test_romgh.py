"""Tests for romgh -- hierarchical omega."""
import numpy as np
from morie.fn.romgh import omega_hierarchical_sub
from morie.fn._containers import ESRes


class TestOmegaHSub:
    def test_basic(self):
        lg = np.array([0.5, 0.6, 0.4, 0.5, 0.3])
        ls = np.array([0.4, 0.3, 0.5, 0.4, 0.6])
        result = omega_hierarchical_sub(lg, ls)
        assert isinstance(result, ESRes)
        assert 0 < result.estimate < 1

    def test_ecv(self):
        lg = np.array([0.7, 0.7, 0.7])
        ls = np.array([0.3, 0.3, 0.3])
        result = omega_hierarchical_sub(lg, ls)
        assert "ECV" in result.extra
        assert result.extra["ECV"] > 0.5
