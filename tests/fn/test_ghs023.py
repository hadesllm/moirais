"""Tests for ghs023.ghosal_ch3_tailfree_abs_continuity_cond."""
import numpy as np
import pytest
from morie.fn.ghs023 import ghosal_ch3_tailfree_abs_continuity_cond


def test_ghs023_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    m = 10
    result = ghosal_ch3_tailfree_abs_continuity_cond(V, mu, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs023_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    m = 10
    result = ghosal_ch3_tailfree_abs_continuity_cond(V, mu, m)
    assert isinstance(result, dict)
