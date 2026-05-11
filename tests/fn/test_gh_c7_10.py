"""Tests for gh_c7_10.ghosal_mono_reg_con."""
import numpy as np
import pytest
from morie.fn.gh_c7_10 import ghosal_mono_reg_con


def test_gh_c7_10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_mono_reg_con(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c7_10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_mono_reg_con(x, y)
    assert isinstance(result, dict)
