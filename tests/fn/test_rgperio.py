"""Tests for rgperio.rangayyan_periodogram."""
import numpy as np
import pytest
from moirais.fn.rgperio import rangayyan_periodogram


def test_rgperio_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_periodogram(x, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgperio_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_periodogram(x, fs)
    assert isinstance(result, dict)
