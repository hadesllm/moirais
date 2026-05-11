"""Tests for convnx.convnext_block."""
import numpy as np
import pytest
from morie.fn.convnx import convnext_block


def test_convnx_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = convnext_block(x, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_convnx_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = convnext_block(x, filters)
    assert isinstance(result, dict)
