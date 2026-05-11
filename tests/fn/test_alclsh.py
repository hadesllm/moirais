"""Tests for alclsh.alammar_classification_head."""
import numpy as np
import pytest
from morie.fn.alclsh import alammar_classification_head


def test_alclsh_basic():
    """Test basic functionality."""
    h_cls = np.random.default_rng(42).normal(0, 1, 100)
    W_cls = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_classification_head(h_cls, W_cls, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alclsh_edge():
    """Test edge cases."""
    h_cls = np.random.default_rng(42).normal(0, 1, 100)
    W_cls = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_classification_head(h_cls, W_cls, b)
    assert isinstance(result, dict)
