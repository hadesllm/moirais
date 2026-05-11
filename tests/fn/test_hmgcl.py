"""Tests for hmgcl.geron_gradient_clipping."""
import numpy as np
import pytest
from morie.fn.hmgcl import geron_gradient_clipping


def test_hmgcl_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    max_norm = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_clipping(grads, max_norm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmgcl_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    max_norm = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_clipping(grads, max_norm)
    assert isinstance(result, dict)
