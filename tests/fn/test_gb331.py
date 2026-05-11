"""Tests for gb331.gibbons_run_lengths_dist."""
import numpy as np
import pytest
from morie.fn.gb331 import gibbons_run_lengths_dist


def test_gb331_basic():
    """Test basic functionality."""
    run_lengths = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_run_lengths_dist(run_lengths, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb331_edge():
    """Test edge cases."""
    run_lengths = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_run_lengths_dist(run_lengths, n1, n2)
    assert isinstance(result, dict)
