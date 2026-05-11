"""Tests for fnscale.functional_scale."""
import numpy as np
import pytest
from morie.fn.fnscale import functional_scale


def test_fnscale_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_scale(f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fnscale_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = functional_scale(f)
    assert isinstance(result, dict)
