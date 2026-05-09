"""Tests for crkbsg.cokriging."""
import numpy as np
import pytest
from moirais.fn.crkbsg import cokriging


def test_crkbsg_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    cross_variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = cokriging(coords, y, z, s_predict, cross_variogram)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crkbsg_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    s_predict = np.random.default_rng(42).normal(0, 1, 100)
    cross_variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = cokriging(coords, y, z, s_predict, cross_variogram)
    assert isinstance(result, dict)
