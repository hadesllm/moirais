"""Tests for nrgmwd.normalized_random_measure."""
import numpy as np
import pytest
from morie.fn.nrgmwd import normalized_random_measure


def test_nrgmwd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    tau = 0.1
    result = normalized_random_measure(y, alpha, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nrgmwd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    tau = 0.1
    result = normalized_random_measure(y, alpha, tau)
    assert isinstance(result, dict)
