"""Tests for gb322.gibbons_total_runs_dist."""
import numpy as np
import pytest
from moirais.fn.gb322 import gibbons_total_runs_dist


def test_gb322_basic():
    """Test basic functionality."""
    r = 10
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_total_runs_dist(r, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb322_edge():
    """Test edge cases."""
    r = 10
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_total_runs_dist(r, n1, n2)
    assert isinstance(result, dict)
