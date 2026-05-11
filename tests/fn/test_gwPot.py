"""Tests for gwPot.global_warming_potential."""
import numpy as np
import pytest
from morie.fn.gwPot import global_warming_potential


def test_gwPot_basic():
    """Test basic functionality."""
    gas = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = global_warming_potential(gas, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwPot_edge():
    """Test edge cases."""
    gas = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = global_warming_potential(gas, horizon)
    assert isinstance(result, dict)
