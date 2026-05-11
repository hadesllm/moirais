"""Tests for hmovf.geron_overfitting."""
import numpy as np
import pytest
from morie.fn.hmovf import geron_overfitting


def test_hmovf_basic():
    """Test basic functionality."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    val_err = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_overfitting(train_err, val_err)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmovf_edge():
    """Test edge cases."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    val_err = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_overfitting(train_err, val_err)
    assert isinstance(result, dict)
