"""Tests for morie.fn.kldiv — KL divergence."""
import numpy as np
import pytest
from morie.fn.kldiv import kl_divergence


class TestKLDivergence:
    def test_same_distribution(self):
        p = np.array([0.5, 0.5])
        res = kl_divergence(p, p)
        assert res.estimate == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        res = kl_divergence(p, q)
        assert res.estimate > 0

    def test_zero_q_raises(self):
        with pytest.raises(ValueError):
            kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
