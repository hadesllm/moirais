"""Tests for hmbntr.geron_batch_normalization."""
import numpy as np
import pytest
from moirais.fn.hmbntr import geron_batch_normalization


def test_hmbntr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_batch_normalization(x, gamma, beta, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbntr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    beta = 0.8
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_batch_normalization(x, gamma, beta, eps)
    assert isinstance(result, dict)
