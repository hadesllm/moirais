"""Tests for otmtxe.ot_matrix_scaling."""
import numpy as np
import pytest
from morie.fn.otmtxe import ot_matrix_scaling


def test_otmtxe_basic():
    """Test basic functionality."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    row_target = np.random.default_rng(42).normal(0, 1, 100)
    col_target = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_matrix_scaling(K, row_target, col_target, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmtxe_edge():
    """Test edge cases."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    row_target = np.random.default_rng(42).normal(0, 1, 100)
    col_target = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_matrix_scaling(K, row_target, col_target, max_iter)
    assert isinstance(result, dict)
