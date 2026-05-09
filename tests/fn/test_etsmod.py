"""Tests for etsmod.ets."""
import numpy as np
import pytest
from moirais.fn.etsmod import ets


def test_etsmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    error = np.random.default_rng(42).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    season = np.random.default_rng(42).normal(0, 1, 100)
    result = ets(y, error, trend, season)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_etsmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    error = np.random.default_rng(42).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    season = np.random.default_rng(42).normal(0, 1, 100)
    result = ets(y, error, trend, season)
    assert isinstance(result, dict)
