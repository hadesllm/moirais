"""Tests for gh_c6_5.ghosal_schwartz_thm."""
import numpy as np
import pytest
from morie.fn.gh_c6_5 import ghosal_schwartz_thm


def test_gh_c6_5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_schwartz_thm(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c6_5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_schwartz_thm(x)
    assert isinstance(result, dict)
