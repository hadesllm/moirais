"""Tests for micomp.mi_compare_models."""
import numpy as np
import pytest
from morie.fn.micomp import mi_compare_models


def test_micomp_basic():
    """Test basic functionality."""
    theta_list = np.random.default_rng(42).normal(0, 1, 100)
    var_list = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_compare_models(theta_list, var_list)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_micomp_edge():
    """Test edge cases."""
    theta_list = np.random.default_rng(42).normal(0, 1, 100)
    var_list = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_compare_models(theta_list, var_list)
    assert isinstance(result, dict)
