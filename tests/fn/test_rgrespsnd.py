"""Tests for rgrespsnd.rangayyan_respiratory_sound."""
import numpy as np
import pytest
from morie.fn.rgrespsnd import rangayyan_respiratory_sound


def test_rgrespsnd_basic():
    """Test basic functionality."""
    resp_sound = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_respiratory_sound(resp_sound, fs, flow)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrespsnd_edge():
    """Test edge cases."""
    resp_sound = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_respiratory_sound(resp_sound, fs, flow)
    assert isinstance(result, dict)
