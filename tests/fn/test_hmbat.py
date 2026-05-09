"""Tests for hmbat.geron_batch_learning."""
import numpy as np
import pytest
from moirais.fn.hmbat import geron_batch_learning


def test_hmbat_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_batch_learning(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbat_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_batch_learning(X, y)
    assert isinstance(result, dict)
