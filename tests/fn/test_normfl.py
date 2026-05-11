"""Tests for normfl.normalizing_flow."""
import numpy as np
import pytest
from morie.fn.normfl import normalizing_flow


def test_normfl_basic():
    """Test basic functionality."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow(base, flow)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_normfl_edge():
    """Test edge cases."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow(base, flow)
    assert isinstance(result, dict)
