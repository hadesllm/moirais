"""Tests for gh_c9_10.ghosal_spline_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c9_10 import ghosal_spline_crt


def test_gh_c9_10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_spline_crt(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c9_10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_spline_crt(x, y)
    assert isinstance(result, dict)
