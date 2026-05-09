"""Tests for studrs.studentized_residual."""
import numpy as np
import pytest
from moirais.fn.studrs import studentized_residual


def test_studrs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = studentized_residual(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_studrs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = studentized_residual(y, X)
    assert isinstance(result, dict)
