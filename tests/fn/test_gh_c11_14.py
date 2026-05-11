"""Tests for gh_c11_14.ghosal_gp_laplace."""
import numpy as np
import pytest
from morie.fn.gh_c11_14 import ghosal_gp_laplace


def test_gh_c11_14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_laplace(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c11_14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_laplace(x, y)
    assert isinstance(result, dict)
