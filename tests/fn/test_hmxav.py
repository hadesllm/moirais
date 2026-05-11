"""Tests for hmxav.geron_glorot_init."""
import numpy as np
import pytest
from morie.fn.hmxav import geron_glorot_init


def test_hmxav_basic():
    """Test basic functionality."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    fan_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_glorot_init(fan_in, fan_out, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmxav_edge():
    """Test edge cases."""
    fan_in = np.random.default_rng(42).normal(0, 1, 100)
    fan_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_glorot_init(fan_in, fan_out, seed)
    assert isinstance(result, dict)
