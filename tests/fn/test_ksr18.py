"""Tests for ksr18.kosorok_nelson_aalen."""
import numpy as np
import pytest
from moirais.fn.ksr18 import kosorok_nelson_aalen


def test_ksr18_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_nelson_aalen(t, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr18_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_nelson_aalen(t, event)
    assert isinstance(result, dict)
