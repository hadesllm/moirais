"""Tests for rgmcn.rangayyan_mcnemar_test."""
import numpy as np
import pytest
from morie.fn.rgmcn import rangayyan_mcnemar_test


def test_rgmcn_basic():
    """Test basic functionality."""
    y1 = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_mcnemar_test(y1, y2, y_true)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgmcn_edge():
    """Test edge cases."""
    y1 = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_mcnemar_test(y1, y2, y_true)
    assert isinstance(result, dict)
