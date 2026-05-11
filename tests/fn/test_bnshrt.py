"""Tests for bnshrt.bound_short_panel."""
import numpy as np
import pytest
from morie.fn.bnshrt import bound_short_panel


def test_bnshrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = bound_short_panel(y, D, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnshrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = bound_short_panel(y, D, time)
    assert isinstance(result, dict)
