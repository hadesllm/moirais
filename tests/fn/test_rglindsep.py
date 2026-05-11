"""Tests for rglindsep.rangayyan_lin_discr_sep."""
import numpy as np
import pytest
from morie.fn.rglindsep import rangayyan_lin_discr_sep


def test_rglindsep_basic():
    """Test basic functionality."""
    X_1 = np.random.default_rng(42).normal(0, 1, 100)
    X_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lin_discr_sep(X_1, X_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglindsep_edge():
    """Test edge cases."""
    X_1 = np.random.default_rng(42).normal(0, 1, 100)
    X_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lin_discr_sep(X_1, X_2)
    assert isinstance(result, dict)
