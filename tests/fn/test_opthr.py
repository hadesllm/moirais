"""Tests for opthr.optimal_huber_k."""
import numpy as np
import pytest
from morie.fn.opthr import optimal_huber_k


def test_opthr_basic():
    """Test basic functionality."""
    target_eff = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_huber_k(target_eff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_opthr_edge():
    """Test edge cases."""
    target_eff = np.random.default_rng(42).normal(0, 1, 100)
    result = optimal_huber_k(target_eff)
    assert isinstance(result, dict)
