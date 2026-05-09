"""Tests for cvxmnr.boyd_minimax."""
import numpy as np
import pytest
from moirais.fn.cvxmnr import boyd_minimax


def test_cvxmnr_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_minimax(f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxmnr_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_minimax(f)
    assert isinstance(result, dict)
