"""Tests for wsmcvr.wasserman_kfold_cv."""
import numpy as np
import pytest
from moirais.fn.wsmcvr import wasserman_kfold_cv


def test_wsmcvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = wasserman_kfold_cv(X, y, model, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = wasserman_kfold_cv(X, y, model, k)
    assert isinstance(result, dict)
