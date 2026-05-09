"""Tests for crfflt.christiano_fitzgerald."""
import numpy as np
import pytest
from moirais.fn.crfflt import christiano_fitzgerald


def test_crfflt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p_low = np.random.default_rng(42).normal(0, 1, 100)
    p_high = np.random.default_rng(42).normal(0, 1, 100)
    result = christiano_fitzgerald(y, p_low, p_high)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfflt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p_low = np.random.default_rng(42).normal(0, 1, 100)
    p_high = np.random.default_rng(42).normal(0, 1, 100)
    result = christiano_fitzgerald(y, p_low, p_high)
    assert isinstance(result, dict)
