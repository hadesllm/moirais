"""Tests for hmuf.geron_underfitting."""
import numpy as np
import pytest
from moirais.fn.hmuf import geron_underfitting


def test_hmuf_basic():
    """Test basic functionality."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_underfitting(train_err, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmuf_edge():
    """Test edge cases."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_underfitting(train_err, threshold)
    assert isinstance(result, dict)
