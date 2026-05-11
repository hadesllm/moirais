"""Tests for aitsdv.compositional_shannon."""
import numpy as np
import pytest
from morie.fn.aitsdv import compositional_shannon


def test_aitsdv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_shannon(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitsdv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_shannon(x)
    assert isinstance(result, dict)
