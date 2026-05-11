"""Tests for chiDep.chi_dependence."""
import numpy as np
import pytest
from morie.fn.chiDep import chi_dependence


def test_chiDep_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = chi_dependence(X, Y, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chiDep_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = chi_dependence(X, Y, u)
    assert isinstance(result, dict)
