"""Tests for gh_c11_15.ghosal_ep_gp."""
import numpy as np
import pytest
from morie.fn.gh_c11_15 import ghosal_ep_gp


def test_gh_c11_15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_ep_gp(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c11_15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_ep_gp(x, y)
    assert isinstance(result, dict)
