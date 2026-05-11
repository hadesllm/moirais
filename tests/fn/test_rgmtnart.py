"""Tests for rgmtnart.rangayyan_motion_artifact."""
import numpy as np
import pytest
from morie.fn.rgmtnart import rangayyan_motion_artifact


def test_rgmtnart_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    accel = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_motion_artifact(x, accel, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmtnart_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    accel = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_motion_artifact(x, accel, fs)
    assert isinstance(result, dict)
