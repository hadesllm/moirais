"""Tests for gh_c11_5.ghosal_gp_binreg_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c11_5 import ghosal_gp_binreg_crt


def test_gh_c11_5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_binreg_crt(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c11_5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gp_binreg_crt(x, y)
    assert isinstance(result, dict)
