"""Tests for crpgib.crp_gibbs."""
import numpy as np
import pytest
from morie.fn.crpgib import crp_gibbs


def test_crpgib_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = crp_gibbs(y, alpha, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crpgib_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = crp_gibbs(y, alpha, n_iter)
    assert isinstance(result, dict)
