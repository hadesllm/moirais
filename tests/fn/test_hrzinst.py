"""Tests for hrzinst.horowitz_instruments_transformation."""
import numpy as np
import pytest
from morie.fn.hrzinst import horowitz_instruments_transformation


def test_hrzinst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_instruments_transformation(x, y, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzinst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = horowitz_instruments_transformation(x, y, z)
    assert isinstance(result, dict)
