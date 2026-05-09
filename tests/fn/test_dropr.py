"""Tests for dropr.dropout_regularization."""
import numpy as np
import pytest
from moirais.fn.dropr import dropout_regularization


def test_dropr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = dropout_regularization(x, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dropr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = dropout_regularization(x, p)
    assert isinstance(result, dict)
