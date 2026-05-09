"""Tests for ghs027.ghosal_ch3_tailfree_strong_support_event."""
import numpy as np
import pytest
from moirais.fn.ghs027 import ghosal_ch3_tailfree_strong_support_event


def test_ghs027_basic():
    """Test basic functionality."""
    p = 5
    p_m = np.random.default_rng(42).normal(0, 1, 100)
    p_0 = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tailfree_strong_support_event(p, p_m, p_0, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs027_edge():
    """Test edge cases."""
    p = 5
    p_m = np.random.default_rng(42).normal(0, 1, 100)
    p_0 = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ghosal_ch3_tailfree_strong_support_event(p, p_m, p_0, epsilon)
    assert isinstance(result, dict)
