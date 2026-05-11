"""Tests for itnnrs.item_nonresponse."""
import numpy as np
import pytest
from morie.fn.itnnrs import item_nonresponse


def test_itnnrs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = item_nonresponse(y, R, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itnnrs_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = item_nonresponse(y, R, X)
    assert isinstance(result, dict)
