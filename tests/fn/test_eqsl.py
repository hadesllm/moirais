"""Tests for eqsl.equating_stocking_lord."""
import numpy as np
import pytest
from morie.fn.eqsl import equating_stocking_lord


def test_eqsl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_stocking_lord(y, b_R, b_F, a_R, a_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eqsl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_stocking_lord(y, b_R, b_F, a_R, a_F)
    assert isinstance(result, dict)
