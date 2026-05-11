"""Tests for survlts.life_table_smoothed."""
import numpy as np
import pytest
from morie.fn.survlts import life_table_smoothed


def test_survlts_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = life_table_smoothed(time, event, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survlts_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = life_table_smoothed(time, event, bandwidth)
    assert isinstance(result, dict)
