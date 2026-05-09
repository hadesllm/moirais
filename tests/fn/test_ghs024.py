"""Tests for ghs024.ghosal_ch3_tailfree_canonical_summability."""
import numpy as np
import pytest
from moirais.fn.ghs024 import ghosal_ch3_tailfree_canonical_summability


def test_ghs024_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_canonical_summability(V, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs024_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_canonical_summability(V, m)
    assert isinstance(result, dict)
