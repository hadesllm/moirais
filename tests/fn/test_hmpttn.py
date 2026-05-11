"""Tests for hmpttn.geron_pytorch_tensor."""
import numpy as np
import pytest
from morie.fn.hmpttn import geron_pytorch_tensor


def test_hmpttn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    device = np.random.default_rng(42).normal(0, 1, 100)
    dtype = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pytorch_tensor(x, device, dtype)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpttn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    device = np.random.default_rng(42).normal(0, 1, 100)
    dtype = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pytorch_tensor(x, device, dtype)
    assert isinstance(result, dict)
