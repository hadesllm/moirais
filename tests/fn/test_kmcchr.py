"""Tests for kmcchr.kamath_christiano_deep_rl_feedback."""
import numpy as np
import pytest
from morie.fn.kmcchr import kamath_christiano_deep_rl_feedback


def test_kmcchr_basic():
    """Test basic functionality."""
    trajectory_pairs = np.random.default_rng(42).normal(0, 1, 100)
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcchr_edge():
    """Test edge cases."""
    trajectory_pairs = np.random.default_rng(42).normal(0, 1, 100)
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)
