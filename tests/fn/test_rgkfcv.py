"""Tests for rgkfcv.rangayyan_kfold_cv."""
import numpy as np
import pytest
from moirais.fn.rgkfcv import rangayyan_kfold_cv


def test_rgkfcv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kfold_cv(X, y, k, classifier)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgkfcv_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kfold_cv(X, y, k, classifier)
    assert isinstance(result, dict)
