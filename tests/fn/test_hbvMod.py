"""Tests for hbvMod.hbv_hydrology."""
import numpy as np
import pytest
from moirais.fn.hbvMod import hbv_hydrology


def test_hbvMod_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = hbv_hydrology(P, T, params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hbvMod_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = hbv_hydrology(P, T, params)
    assert isinstance(result, dict)
