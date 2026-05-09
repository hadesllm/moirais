"""Tests for jorelm.joseph_relative_mae."""
import numpy as np
import pytest
from moirais.fn.jorelm import joseph_relative_mae


def test_jorelm_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    y_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_relative_mae(y_true, y_pred, y_baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jorelm_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    y_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_relative_mae(y_true, y_pred, y_baseline)
    assert isinstance(result, dict)
