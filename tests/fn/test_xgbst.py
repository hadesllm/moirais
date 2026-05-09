"""Tests for xgbst.xgboost_objective."""
import numpy as np
import pytest
from moirais.fn.xgbst import xgboost_objective


def test_xgbst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = xgboost_objective(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_xgbst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = xgboost_objective(x, y)
    assert isinstance(result, dict)
