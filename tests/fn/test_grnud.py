"""Tests for grnud.geron_numerical_differentiation."""
import numpy as np
import pytest
from morie.fn.grnud import geron_numerical_differentiation


def test_grnud_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_numerical_differentiation(f, x, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grnud_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = geron_numerical_differentiation(f, x, h)
    assert isinstance(result, dict)
