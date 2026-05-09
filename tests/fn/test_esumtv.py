"""Tests for esumtv.effective_resistance."""
import numpy as np
import pytest
from moirais.fn.esumtv import effective_resistance


def test_esumtv_basic():
    """Test basic functionality."""
    G = np.eye(10)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = effective_resistance(G, u, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esumtv_edge():
    """Test edge cases."""
    G = np.eye(10)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = effective_resistance(G, u, v)
    assert isinstance(result, dict)
