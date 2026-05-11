"""Tests for otdwd.ot_doubly_stoch_proj."""
import numpy as np
import pytest
from morie.fn.otdwd import ot_doubly_stoch_proj


def test_otdwd_basic():
    """Test basic functionality."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_doubly_stoch_proj(K, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otdwd_edge():
    """Test edge cases."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_doubly_stoch_proj(K, max_iter)
    assert isinstance(result, dict)
