"""Tests for gb332.gibbons_type1_run_lengths."""
import numpy as np
import pytest
from morie.fn.gb332 import gibbons_type1_run_lengths


def test_gb332_basic():
    """Test basic functionality."""
    run_lengths = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_run_lengths(run_lengths, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb332_edge():
    """Test edge cases."""
    run_lengths = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_run_lengths(run_lengths, n1, n2)
    assert isinstance(result, dict)
