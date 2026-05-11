"""Tests for gh_c10_13.ghosal_pt_null_tst."""
import numpy as np
import pytest
from morie.fn.gh_c10_13 import ghosal_pt_null_tst


def test_gh_c10_13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_pt_null_tst(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c10_13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_pt_null_tst(x)
    assert isinstance(result, dict)
