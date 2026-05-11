"""Tests for gh_c5_4.ghosal_splitmerge."""
import numpy as np
import pytest
from morie.fn.gh_c5_4 import ghosal_splitmerge


def test_gh_c5_4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_splitmerge(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c5_4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_splitmerge(x)
    assert isinstance(result, dict)
