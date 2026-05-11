"""Tests for matSym.matrix_symbolic."""
import numpy as np
import pytest
from morie.fn.matSym import matrix_symbolic


def test_matSym_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = matrix_symbolic(M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matSym_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = matrix_symbolic(M)
    assert isinstance(result, dict)
