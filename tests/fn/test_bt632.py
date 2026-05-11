"""Tests for bt632.boot_632_estimator."""
import numpy as np
import pytest
from morie.fn.bt632 import boot_632_estimator


def test_bt632_basic():
    """Test basic functionality."""
    err_app = np.random.default_rng(42).normal(0, 1, 100)
    err_oob = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_632_estimator(err_app, err_oob)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bt632_edge():
    """Test edge cases."""
    err_app = np.random.default_rng(42).normal(0, 1, 100)
    err_oob = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_632_estimator(err_app, err_oob)
    assert isinstance(result, dict)
