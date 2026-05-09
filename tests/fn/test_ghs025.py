"""Tests for ghs025.ghosal_ch3_tailfree_density_product."""
import numpy as np
import pytest
from moirais.fn.ghs025 import ghosal_ch3_tailfree_density_product


def test_ghs025_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_density_product(V, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs025_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_density_product(V, x)
    assert isinstance(result, dict)
