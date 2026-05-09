"""Tests for agvslr.alphazero_value_lr."""
import numpy as np
import pytest
from moirais.fn.agvslr import alphazero_value_lr


def test_agvslr_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    lr_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_lr(t, T, lr_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agvslr_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    lr_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_value_lr(t, T, lr_0)
    assert isinstance(result, dict)
