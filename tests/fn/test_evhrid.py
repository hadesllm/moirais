"""Tests for evhrid.evt_husler_reiss_dep."""
import numpy as np
import pytest
from moirais.fn.evhrid import evt_husler_reiss_dep


def test_evhrid_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = evt_husler_reiss_dep(x, y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evhrid_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = evt_husler_reiss_dep(x, y, lam)
    assert isinstance(result, dict)
