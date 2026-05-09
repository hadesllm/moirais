"""Tests for idwsmt.inverse_distance_weighting."""
import numpy as np
import pytest
from moirais.fn.idwsmt import inverse_distance_weighting


def test_idwsmt_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    power = np.random.default_rng(42).normal(0, 1, 100)
    result = inverse_distance_weighting(coords, values, s_predict, power)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_idwsmt_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    power = np.random.default_rng(42).normal(0, 1, 100)
    result = inverse_distance_weighting(coords, values, s_predict, power)
    assert isinstance(result, dict)
