"""Tests for btbg.boot_bagging_predict."""
import numpy as np
import pytest
from moirais.fn.btbg import boot_bagging_predict


def test_btbg_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X_new = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_bagging_predict(models, X_new, kind)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btbg_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X_new = np.random.default_rng(42).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_bagging_predict(models, X_new, kind)
    assert isinstance(result, dict)
