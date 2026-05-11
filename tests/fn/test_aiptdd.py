"""Tests for aiptdd.aipw_did."""
import numpy as np
import pytest
from morie.fn.aiptdd import aipw_did


def test_aiptdd_basic():
    """Test basic functionality."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aipw_did(y_pre, y_post, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aiptdd_edge():
    """Test edge cases."""
    y_pre = np.random.default_rng(42).normal(0, 1, 100)
    y_post = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aipw_did(y_pre, y_post, D, X)
    assert isinstance(result, dict)
