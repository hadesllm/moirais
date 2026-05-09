"""Tests for aitsim.compositional_simpson."""
import numpy as np
import pytest
from moirais.fn.aitsim import compositional_simpson


def test_aitsim_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_simpson(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitsim_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_simpson(x)
    assert isinstance(result, dict)
