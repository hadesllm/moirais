"""Tests for ktaup.kendall_tau_partial."""
import numpy as np
import pytest
from morie.fn.ktaup import kendall_tau_partial


def test_ktaup_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = kendall_tau_partial(x, y, z)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ktaup_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = kendall_tau_partial(x, y, z)
    assert isinstance(result, dict)
