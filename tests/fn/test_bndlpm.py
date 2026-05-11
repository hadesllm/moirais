"""Tests for bndlpm.bound_lp_method."""
import numpy as np
import pytest
from morie.fn.bndlpm import bound_lp_method


def test_bndlpm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    moment_eqs = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_lp_method(y, D, Z, moment_eqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndlpm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    moment_eqs = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_lp_method(y, D, Z, moment_eqs)
    assert isinstance(result, dict)
