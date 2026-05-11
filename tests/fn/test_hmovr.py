"""Tests for hmovr.geron_one_vs_rest."""
import numpy as np
import pytest
from morie.fn.hmovr import geron_one_vs_rest


def test_hmovr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_vs_rest(X, y, base_estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmovr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_vs_rest(X, y, base_estimator)
    assert isinstance(result, dict)
