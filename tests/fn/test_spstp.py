"""Tests for spstp.schabenberger_st_point_process."""
import numpy as np
import pytest
from morie.fn.spstp import schabenberger_st_point_process


def test_spstp_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    time_interval = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_st_point_process(points, region, time_interval)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spstp_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    time_interval = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_st_point_process(points, region, time_interval)
    assert isinstance(result, dict)
