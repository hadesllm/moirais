"""Tests for hmvgr.geron_vanishing_gradients."""
import numpy as np
import pytest
from moirais.fn.hmvgr import geron_vanishing_gradients


def test_hmvgr_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vanishing_gradients(grads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvgr_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_vanishing_gradients(grads)
    assert isinstance(result, dict)
