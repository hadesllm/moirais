"""Tests for mtxrl.matrix_game."""
import numpy as np
import pytest
from moirais.fn.mtxrl import matrix_game


def test_mtxrl_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = matrix_game(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtxrl_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = matrix_game(A)
    assert isinstance(result, dict)
