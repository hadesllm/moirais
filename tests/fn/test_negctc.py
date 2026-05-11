"""Tests for negctc.negative_control_outcome."""
import numpy as np
import pytest
from morie.fn.negctc import negative_control_outcome


def test_negctc_basic():
    """Test basic functionality."""
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = negative_control_outcome(y_neg, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_negctc_edge():
    """Test edge cases."""
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = negative_control_outcome(y_neg, D, X)
    assert isinstance(result, dict)
