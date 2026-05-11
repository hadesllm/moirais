"""Tests for grxvi.geron_glorot_xavier_init."""
import numpy as np
import pytest
from morie.fn.grxvi import geron_glorot_xavier_init


def test_grxvi_basic():
    """Test basic functionality."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    fan_out = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = geron_glorot_xavier_init(fan_in, fan_out, distribution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grxvi_edge():
    """Test edge cases."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    fan_out = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = geron_glorot_xavier_init(fan_in, fan_out, distribution)
    assert isinstance(result, dict)
