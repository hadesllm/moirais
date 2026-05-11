"""Tests for tmlcom.tmle_compositional."""
import numpy as np
import pytest
from morie.fn.tmlcom import tmle_compositional


def test_tmlcom_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    composition = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_compositional(y, composition, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcom_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    composition = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_compositional(y, composition, X)
    assert isinstance(result, dict)
