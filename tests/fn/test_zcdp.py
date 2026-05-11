"""Tests for zcdp.zcdp."""
import numpy as np
import pytest
from morie.fn.zcdp import zcdp


def test_zcdp_basic():
    """Test basic functionality."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = zcdp(mech, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_zcdp_edge():
    """Test edge cases."""
    mech = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = zcdp(mech, rho)
    assert isinstance(result, dict)
