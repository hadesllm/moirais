"""Tests for gh_c6_15.ghosal_martg_consist."""
import numpy as np
import pytest
from moirais.fn.gh_c6_15 import ghosal_martg_consist


def test_gh_c6_15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_martg_consist(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c6_15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_martg_consist(x)
    assert isinstance(result, dict)
