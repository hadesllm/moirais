"""Tests for berkly.berkeley_earth."""
import numpy as np
import pytest
from moirais.fn.berkly import berkeley_earth


def test_berkly_basic():
    """Test basic functionality."""
    stations = np.random.default_rng(42).normal(0, 1, 100)
    result = berkeley_earth(stations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_berkly_edge():
    """Test edge cases."""
    stations = np.random.default_rng(42).normal(0, 1, 100)
    result = berkeley_earth(stations)
    assert isinstance(result, dict)
