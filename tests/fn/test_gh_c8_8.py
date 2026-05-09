"""Tests for gh_c8_8.ghosal_gauss_reg_crt."""
import numpy as np
import pytest
from moirais.fn.gh_c8_8 import ghosal_gauss_reg_crt


def test_gh_c8_8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gauss_reg_crt(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gh_c8_8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ghosal_gauss_reg_crt(x, y)
    assert isinstance(result, dict)
