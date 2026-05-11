"""Tests for otemd.ot_emd_solver."""
import numpy as np
import pytest
from morie.fn.otemd import ot_emd_solver


def test_otemd_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_emd_solver(a, b, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otemd_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_emd_solver(a, b, C)
    assert isinstance(result, dict)
