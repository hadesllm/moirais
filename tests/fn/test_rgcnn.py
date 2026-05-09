"""Tests for rgcnn.rangayyan_cnn_signal."""
import numpy as np
import pytest
from moirais.fn.rgcnn import rangayyan_cnn_signal


def test_rgcnn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    kernel_sizes = np.random.default_rng(42).normal(0, 1, 100)
    n_classes = 3
    result = rangayyan_cnn_signal(x, filters, kernel_sizes, n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcnn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    kernel_sizes = np.random.default_rng(42).normal(0, 1, 100)
    n_classes = 3
    result = rangayyan_cnn_signal(x, filters, kernel_sizes, n_classes)
    assert isinstance(result, dict)
