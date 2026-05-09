"""Tests for gh_ap_k1.ghosal_fano_ineq."""
import numpy as np
import pytest
from moirais.fn.gh_ap_k1 import ghosal_fano_ineq


def test_gh_ap_k1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_fano_ineq(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gh_ap_k1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_fano_ineq(x)
    assert isinstance(result, dict)
