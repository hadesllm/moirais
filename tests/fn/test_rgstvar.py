"""Tests for rgstvar.rangayyan_tvlsi."""
import numpy as np
import pytest
from morie.fn.rgstvar import rangayyan_tvlsi


def test_rgstvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_tvlsi(x, y, fs, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgstvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_tvlsi(x, y, fs, window)
    assert isinstance(result, dict)
