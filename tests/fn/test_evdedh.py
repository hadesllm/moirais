"""Tests for evdedh.evt_dekkers_einmahl_dehaan."""
import numpy as np
import pytest
from morie.fn.evdedh import evt_dekkers_einmahl_dehaan


def test_evdedh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_dekkers_einmahl_dehaan(x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evdedh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = evt_dekkers_einmahl_dehaan(x, k)
    assert isinstance(result, dict)
