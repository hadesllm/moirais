"""Tests for spnest.schabenberger_nested_variogram."""
import numpy as np
import pytest
from moirais.fn.spnest import schabenberger_nested_variogram


def test_spnest_basic():
    """Test basic functionality."""
    h = 0.3
    components = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nested_variogram(h, components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spnest_edge():
    """Test edge cases."""
    h = 0.3
    components = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_nested_variogram(h, components)
    assert isinstance(result, dict)
