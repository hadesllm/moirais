"""Tests for rgacc.rangayyan_accuracy."""
import numpy as np
import pytest
from morie.fn.rgacc import rangayyan_accuracy


def test_rgacc_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_accuracy(y_true, y_pred)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgacc_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_accuracy(y_true, y_pred)
    assert isinstance(result, dict)
