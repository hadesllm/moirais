"""Tests for gb321.gibbons_runs_joint_dist."""
import numpy as np
import pytest
from morie.fn.gb321 import gibbons_runs_joint_dist


def test_gb321_basic():
    """Test basic functionality."""
    r1 = np.random.default_rng(42).normal(0, 1, 100)
    r2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_joint_dist(r1, r2, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb321_edge():
    """Test edge cases."""
    r1 = np.random.default_rng(42).normal(0, 1, 100)
    r2 = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_joint_dist(r1, r2, n1, n2)
    assert isinstance(result, dict)
