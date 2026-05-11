"""Tests for hmsgdc.geron_sgd_classifier."""
import numpy as np
import pytest
from morie.fn.hmsgdc import geron_sgd_classifier


def test_hmsgdc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_sgd_classifier(X, y, lr, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsgdc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_sgd_classifier(X, y, lr, n_iter)
    assert isinstance(result, dict)
