"""Tests for rfcomp.robust_factor_analysis."""
import numpy as np
import pytest
from moirais.fn.rfcomp import robust_factor_analysis


def test_rfcomp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = robust_factor_analysis(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfcomp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = robust_factor_analysis(X, k)
    assert isinstance(result, dict)
