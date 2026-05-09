"""Tests for fn/ess_s.py -- Effective sample size (survey/sampling)."""
import numpy as np

from moirais.fn.ess_s import ess_s, effective_sample_size


def test_ess_s_equal_weights():
    """Equal weights should give ESS = n."""
    w = np.ones(100)
    result = ess_s(w)
    assert abs(result - 100.0) < 1e-10


def test_ess_s_unequal_weights():
    """Unequal weights should give ESS < n."""
    w = np.array([1.0, 1.0, 10.0])
    result = effective_sample_size(w)
    assert result < 3.0
    assert result > 0.0


def test_ess_s_single_large_weight():
    """One dominant weight should push ESS close to 1."""
    w = np.array([0.01, 0.01, 0.01, 100.0])
    result = ess_s(w)
    assert result < 2.0
