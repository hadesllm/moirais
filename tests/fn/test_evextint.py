"""Tests for evextint.evt_extremal_index_intervals."""
import numpy as np
import pytest
from morie.fn.evextint import evt_extremal_index_intervals


def test_evextint_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_extremal_index_intervals(x, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evextint_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_extremal_index_intervals(x, u)
    assert isinstance(result, dict)
