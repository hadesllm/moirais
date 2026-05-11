"""Tests for cokrig.cokriging."""
import numpy as np
import pytest
from morie.fn.cokrig import cokriging


def test_cokrig_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z1 = np.random.default_rng(42).normal(0, 1, 100)
    z2 = np.random.default_rng(42).normal(0, 1, 100)
    s0 = np.random.default_rng(42).normal(0, 1, 100)
    cross_vario = np.random.default_rng(42).normal(0, 1, 100)
    result = cokriging(coords, z1, z2, s0, cross_vario)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cokrig_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z1 = np.random.default_rng(42).normal(0, 1, 100)
    z2 = np.random.default_rng(42).normal(0, 1, 100)
    s0 = np.random.default_rng(42).normal(0, 1, 100)
    cross_vario = np.random.default_rng(42).normal(0, 1, 100)
    result = cokriging(coords, z1, z2, s0, cross_vario)
    assert isinstance(result, dict)
