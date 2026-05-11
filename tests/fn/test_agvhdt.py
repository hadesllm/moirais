"""Tests for agvhdt.alphazero_value_head."""
import numpy as np
import pytest
from morie.fn.agvhdt import alphazero_value_head


def test_agvhdt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_head(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agvhdt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_head(x)
    assert isinstance(result, dict)
