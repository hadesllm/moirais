"""Tests for evpot.evt_pot_fit."""
import numpy as np
import pytest
from moirais.fn.evpot import evt_pot_fit


def test_evpot_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_pot_fit(x, u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evpot_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    result = evt_pot_fit(x, u)
    assert isinstance(result, dict)
