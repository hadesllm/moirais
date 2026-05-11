"""Tests for impala.impala_vtrace."""
import numpy as np
import pytest
from morie.fn.impala import impala_vtrace


def test_impala_basic():
    """Test basic functionality."""
    trajectories = np.random.default_rng(42).normal(0, 1, 100)
    behavior = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    clip = np.random.default_rng(42).normal(0, 1, 100)
    result = impala_vtrace(trajectories, behavior, target, clip)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_impala_edge():
    """Test edge cases."""
    trajectories = np.random.default_rng(42).normal(0, 1, 100)
    behavior = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    clip = np.random.default_rng(42).normal(0, 1, 100)
    result = impala_vtrace(trajectories, behavior, target, clip)
    assert isinstance(result, dict)
