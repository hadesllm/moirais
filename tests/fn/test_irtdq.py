"""Tests for irtdq.irt_quadratic_utility."""
import numpy as np
import pytest
from morie.fn.irtdq import irt_quadratic_utility


def test_irtdq_basic():
    """Test basic functionality."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    vote_position = np.random.default_rng(42).normal(0, 1, 100)
    discrimination = np.random.default_rng(42).normal(0, 1, 100)
    result = irt_quadratic_utility(ideal_point, vote_position, discrimination)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtdq_edge():
    """Test edge cases."""
    ideal_point = np.random.default_rng(42).normal(0, 1, 100)
    vote_position = np.random.default_rng(42).normal(0, 1, 100)
    discrimination = np.random.default_rng(42).normal(0, 1, 100)
    result = irt_quadratic_utility(ideal_point, vote_position, discrimination)
    assert isinstance(result, dict)
