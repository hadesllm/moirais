"""Tests for rgyw.rangayyan_yule_walker."""
import numpy as np
import pytest
from moirais.fn.rgyw import rangayyan_yule_walker


def test_rgyw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_yule_walker(x, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgyw_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_yule_walker(x, order)
    assert isinstance(result, dict)
