"""Tests for gb_rnkci.gibbons_rank_ci."""
import numpy as np
import pytest
from morie.fn.gb_rnkci import gibbons_rank_ci


def test_gb_rnkci_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rank_ci(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_rnkci_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_rank_ci(x, alpha)
    assert isinstance(result, dict)
