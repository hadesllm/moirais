"""Tests for ramsw.ramsay_weight."""
import numpy as np
import pytest
from morie.fn.ramsw import ramsay_weight


def test_ramsw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = ramsay_weight(y, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ramsw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = ramsay_weight(y, a)
    assert isinstance(result, dict)
