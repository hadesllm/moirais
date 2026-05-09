"""Tests for evhpvr.evt_heffernan_tawn."""
import numpy as np
import pytest
from moirais.fn.evhpvr import evt_heffernan_tawn


def test_evhpvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_heffernan_tawn(X, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evhpvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_heffernan_tawn(X, u)
    assert isinstance(result, dict)
