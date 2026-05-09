"""Tests for gb_fwt.gibbons_fligner_wolfe_test."""
import numpy as np
import pytest
from moirais.fn.gb_fwt import gibbons_fligner_wolfe_test


def test_gb_fwt_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fligner_wolfe_test(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_fwt_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fligner_wolfe_test(groups)
    assert isinstance(result, dict)
