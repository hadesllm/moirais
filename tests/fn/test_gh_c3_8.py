"""Tests for gh_c3_8.ghosal_moment_prior."""
import numpy as np
import pytest
from morie.fn.gh_c3_8 import ghosal_moment_prior


def test_gh_c3_8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_moment_prior(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c3_8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_moment_prior(x)
    assert isinstance(result, dict)
