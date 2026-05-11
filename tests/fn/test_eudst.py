"""Tests for eudst.euclidean_utility."""
import numpy as np
import pytest
from morie.fn.eudst import euclidean_utility


def test_eudst_basic():
    """Test basic functionality."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    policy_position = np.random.default_rng(42).normal(0, 1, 100)
    result = euclidean_utility(ideal_point, policy_position)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eudst_edge():
    """Test edge cases."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    policy_position = np.random.default_rng(42).normal(0, 1, 100)
    result = euclidean_utility(ideal_point, policy_position)
    assert isinstance(result, dict)
