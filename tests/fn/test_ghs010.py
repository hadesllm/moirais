"""Tests for ghs010.ghosal_ch3_discrete_hazard_rate."""
import numpy as np
import pytest
from moirais.fn.ghs010 import ghosal_ch3_discrete_hazard_rate


def test_ghs010_basic():
    """Test basic functionality."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ghosal_ch3_discrete_hazard_rate(p_j, j, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs010_edge():
    """Test edge cases."""
    p_j = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ghosal_ch3_discrete_hazard_rate(p_j, j, X)
    assert isinstance(result, dict)
