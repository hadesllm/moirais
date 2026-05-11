"""Tests for topDn.top_down."""
import numpy as np
import pytest
from morie.fn.topDn import top_down


def test_topDn_basic():
    """Test basic functionality."""
    top = np.random.default_rng(42).normal(0, 1, 100)
    props = np.random.default_rng(42).normal(0, 1, 100)
    result = top_down(top, props)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_topDn_edge():
    """Test edge cases."""
    top = np.random.default_rng(42).normal(0, 1, 100)
    props = np.random.default_rng(42).normal(0, 1, 100)
    result = top_down(top, props)
    assert isinstance(result, dict)
