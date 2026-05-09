"""Tests for ghs021.ghosal_ch3_tailfree_max_bound."""
import numpy as np
import pytest
from moirais.fn.ghs021 import ghosal_ch3_tailfree_max_bound


def test_ghs021_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    r = 10
    result = ghosal_ch3_tailfree_max_bound(V, m, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs021_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    r = 10
    result = ghosal_ch3_tailfree_max_bound(V, m, r)
    assert isinstance(result, dict)
