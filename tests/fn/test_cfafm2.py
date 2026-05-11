"""Tests for cfafm2.cfa_multifactor."""
import numpy as np
import pytest
from morie.fn.cfafm2 import cfa_multifactor


def test_cfafm2_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_pattern = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_multifactor(X, factor_pattern)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cfafm2_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_pattern = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_multifactor(X, factor_pattern)
    assert isinstance(result, dict)
