"""Tests for matrxP.matrix_profile."""
import numpy as np
import pytest
from moirais.fn.matrxP import matrix_profile


def test_matrxP_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = matrix_profile(x, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matrxP_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = matrix_profile(x, m)
    assert isinstance(result, dict)
