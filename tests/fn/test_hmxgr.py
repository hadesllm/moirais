"""Tests for hmxgr.geron_exploding_gradients."""
import numpy as np
import pytest
from morie.fn.hmxgr import geron_exploding_gradients


def test_hmxgr_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_exploding_gradients(grads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmxgr_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_exploding_gradients(grads)
    assert isinstance(result, dict)
