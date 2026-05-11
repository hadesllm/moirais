"""Tests for gh_wn_gauss_pr.ghosal_white_noise_gauss_prior."""
import numpy as np
import pytest
from morie.fn.gh_wn_gauss_pr import ghosal_white_noise_gauss_prior


def test_gh_wn_gauss_pr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_white_noise_gauss_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_wn_gauss_pr_edge():
    """Test edge cases."""
    result = ghosal_white_noise_gauss_prior(np.array([42.0]))
    assert result['n'] == 1
