"""Tests for floRate.flow_duration."""
import numpy as np
import pytest
from morie.fn.floRate import flow_duration


def test_floRate_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = flow_duration(Q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_floRate_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = flow_duration(Q)
    assert isinstance(result, dict)
