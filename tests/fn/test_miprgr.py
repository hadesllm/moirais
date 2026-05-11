"""Tests for miprgr.mip_branch_bound."""
import numpy as np
import pytest
from morie.fn.miprgr import mip_branch_bound


def test_miprgr_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    integer_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = mip_branch_bound(c, A, b, integer_indices)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miprgr_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    integer_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = mip_branch_bound(c, A, b, integer_indices)
    assert isinstance(result, dict)
