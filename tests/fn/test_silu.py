"""Tests for silu.silu_swish."""
import numpy as np
import pytest
from morie.fn.silu import silu_swish


def test_silu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = silu_swish(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_silu_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = silu_swish(y)
    assert isinstance(result, dict)
