"""Tests for grgcl.geron_gradient_clipping."""
import numpy as np
import pytest
from moirais.fn.grgcl import geron_gradient_clipping


def test_grgcl_basic():
    """Test basic functionality."""
    gradients = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_clipping(gradients, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgcl_edge():
    """Test edge cases."""
    gradients = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_clipping(gradients, c)
    assert isinstance(result, dict)
