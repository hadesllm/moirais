"""Tests for fwxF.fire_weather_index."""
import numpy as np
import pytest
from morie.fn.fwxF import fire_weather_index


def test_fwxF_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    RH = np.random.default_rng(42).normal(0, 1, 100)
    wind = np.random.default_rng(42).normal(0, 1, 100)
    precip = np.random.default_rng(42).normal(0, 1, 100)
    result = fire_weather_index(T, RH, wind, precip)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fwxF_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    RH = np.random.default_rng(42).normal(0, 1, 100)
    wind = np.random.default_rng(42).normal(0, 1, 100)
    precip = np.random.default_rng(42).normal(0, 1, 100)
    result = fire_weather_index(T, RH, wind, precip)
    assert isinstance(result, dict)
