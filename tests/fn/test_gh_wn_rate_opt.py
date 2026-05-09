"""Tests for gh_wn_rate_opt.ghosal_white_noise_optimal_rate."""
import numpy as np
import pytest
from moirais.fn.gh_wn_rate_opt import ghosal_white_noise_optimal_rate


def test_gh_wn_rate_opt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_white_noise_optimal_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_wn_rate_opt_edge():
    """Test edge cases."""
    result = ghosal_white_noise_optimal_rate(np.array([42.0]))
    assert result['n'] == 1
