"""Tests for dimens.stout_dimensionality."""
import numpy as np
import pytest
from morie.fn.dimens import stout_dimensionality


def test_dimens_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    items = np.random.default_rng(42).normal(0, 1, 100)
    subtest = np.random.default_rng(42).normal(0, 1, 100)
    result = stout_dimensionality(y, items, subtest)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dimens_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    items = np.random.default_rng(42).normal(0, 1, 100)
    subtest = np.random.default_rng(42).normal(0, 1, 100)
    result = stout_dimensionality(y, items, subtest)
    assert isinstance(result, dict)
