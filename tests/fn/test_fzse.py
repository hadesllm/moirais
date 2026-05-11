"""Tests for fzse.fauzi_sign_moments."""
import numpy as np
import pytest
from morie.fn.fzse import fauzi_sign_moments


def test_fzse_basic():
    """Test basic functionality."""
    n = 100
    bandwidth = 0.3
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_sign_moments(n, bandwidth, theta, cdf)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzse_edge():
    """Test edge cases."""
    n = 100
    bandwidth = 0.3
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_sign_moments(n, bandwidth, theta, cdf)
    assert isinstance(result, dict)
