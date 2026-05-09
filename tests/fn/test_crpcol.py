"""Tests for crpcol.crp_collapsed."""
import numpy as np
import pytest
from moirais.fn.crpcol import crp_collapsed


def test_crpcol_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = crp_collapsed(y, alpha, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crpcol_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    n_iter = 50
    result = crp_collapsed(y, alpha, n_iter)
    assert isinstance(result, dict)
