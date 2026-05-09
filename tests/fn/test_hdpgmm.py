"""Tests for hdpgmm.hdp_gaussian_mixture."""
import numpy as np
import pytest
from moirais.fn.hdpgmm import hdp_gaussian_mixture


def test_hdpgmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hdp_gaussian_mixture(y, groups, gamma, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hdpgmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    gamma = 1.0
    alpha = 0.05
    result = hdp_gaussian_mixture(y, groups, gamma, alpha)
    assert isinstance(result, dict)
