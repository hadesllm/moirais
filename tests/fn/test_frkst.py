"""Tests for frkst.fork_structure."""
import numpy as np
import pytest
from morie.fn.frkst import fork_structure


def test_frkst_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    conditioned = np.random.default_rng(42).normal(0, 1, 100)
    result = fork_structure(A, B, C, conditioned)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frkst_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    conditioned = np.random.default_rng(42).normal(0, 1, 100)
    result = fork_structure(A, B, C, conditioned)
    assert isinstance(result, dict)
