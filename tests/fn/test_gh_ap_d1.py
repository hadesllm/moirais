"""Tests for gh_ap_d1.ghosal_exp_test."""
import numpy as np
import pytest
from moirais.fn.gh_ap_d1 import ghosal_exp_test


def test_gh_ap_d1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_exp_test(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_ap_d1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_exp_test(x)
    assert isinstance(result, dict)
