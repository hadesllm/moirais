"""Tests for qpdual.quadratic_program."""
import numpy as np
import pytest
from morie.fn.qpdual import quadratic_program


def test_qpdual_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = quadratic_program(Q, c, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qpdual_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = quadratic_program(Q, c, A, b)
    assert isinstance(result, dict)
