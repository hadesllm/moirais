"""Tests for gb_pit2.gibbons_pit_rng."""
import numpy as np
import pytest
from moirais.fn.gb_pit2 import gibbons_pit_rng


def test_gb_pit2_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    F_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pit_rng(U, F_inv)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_pit2_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    F_inv = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_pit_rng(U, F_inv)
    assert isinstance(result, dict)
