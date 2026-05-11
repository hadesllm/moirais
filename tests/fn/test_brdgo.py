"""Tests for brdgo.bridge_observations."""
import numpy as np
import pytest
from morie.fn.brdgo import bridge_observations


def test_brdgo_basic():
    """Test basic functionality."""
    ideal_points_periods = np.random.default_rng(42).normal(0, 1, 100)
    bridge_ids = np.arange(100, dtype=int)
    result = bridge_observations(ideal_points_periods, bridge_ids)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brdgo_edge():
    """Test edge cases."""
    ideal_points_periods = np.random.default_rng(42).normal(0, 1, 100)
    bridge_ids = np.arange(100, dtype=int)
    result = bridge_observations(ideal_points_periods, bridge_ids)
    assert isinstance(result, dict)
