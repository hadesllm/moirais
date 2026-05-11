"""Tests for grfcn.geron_fcn_upsample."""
import numpy as np
import pytest
from morie.fn.grfcn import geron_fcn_upsample


def test_grfcn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fcn_upsample(X, W, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grfcn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fcn_upsample(X, W, stride)
    assert isinstance(result, dict)
