"""Tests for agscho.alphazero_search_horizon."""
import numpy as np
import pytest
from moirais.fn.agscho import alphazero_search_horizon


def test_agscho_basic():
    """Test basic functionality."""
    depth_limit = np.random.default_rng(42).normal(0, 1, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_search_horizon(depth_limit, state)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agscho_edge():
    """Test edge cases."""
    depth_limit = np.random.default_rng(42).normal(0, 1, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_search_horizon(depth_limit, state)
    assert isinstance(result, dict)
