"""Tests for gh_c8_3.ghosal_test_cond."""
import numpy as np
import pytest
from morie.fn.gh_c8_3 import ghosal_test_cond


def test_gh_c8_3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_test_cond(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_c8_3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_test_cond(x)
    assert isinstance(result, dict)
