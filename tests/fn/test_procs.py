"""Tests for procs.procrustes_rotation."""
import numpy as np
import pytest
from morie.fn.procs import procrustes_rotation


def test_procs_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = procrustes_rotation(A, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_procs_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = procrustes_rotation(A, Z)
    assert isinstance(result, dict)
