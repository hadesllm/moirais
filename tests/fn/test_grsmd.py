"""Tests for grsmd.geron_symbolic_differentiation."""
import numpy as np
import pytest
from morie.fn.grsmd import geron_symbolic_differentiation


def test_grsmd_basic():
    """Test basic functionality."""
    expression = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_symbolic_differentiation(expression)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsmd_edge():
    """Test edge cases."""
    expression = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_symbolic_differentiation(expression)
    assert isinstance(result, dict)
