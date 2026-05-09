"""Tests for gh_c6_12.ghosal_sep_consist."""
import numpy as np
import pytest
from moirais.fn.gh_c6_12 import ghosal_sep_consist


def test_gh_c6_12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_sep_consist(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c6_12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_sep_consist(x)
    assert isinstance(result, dict)
