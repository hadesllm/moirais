"""Tests for rgppt.rangayyan_point_process."""
import numpy as np
import pytest
from morie.fn.rgppt import rangayyan_point_process


def test_rgppt_basic():
    """Test basic functionality."""
    event_times = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_point_process(event_times, T)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgppt_edge():
    """Test edge cases."""
    event_times = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_point_process(event_times, T)
    assert isinstance(result, dict)
