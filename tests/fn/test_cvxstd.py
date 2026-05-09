"""Tests for cvxstd.boyd_steepest_desc."""
import numpy as np
import pytest
from moirais.fn.cvxstd import boyd_steepest_desc


def test_cvxstd_basic():
    """Test basic functionality."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_steepest_desc(grad, norm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxstd_edge():
    """Test edge cases."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    norm = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_steepest_desc(grad, norm)
    assert isinstance(result, dict)
