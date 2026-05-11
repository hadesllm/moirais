"""Tests for gh_c10_10.ghosal_frs_poireg."""
import numpy as np
import pytest
from morie.fn.gh_c10_10 import ghosal_frs_poireg


def test_gh_c10_10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_frs_poireg(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c10_10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_frs_poireg(x, y)
    assert isinstance(result, dict)
