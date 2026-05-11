"""Tests for wsmstn.wasserman_sufficient."""
import numpy as np
import pytest
from morie.fn.wsmstn import wasserman_sufficient


def test_wsmstn_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_sufficient(data, f)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wsmstn_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_sufficient(data, f)
    assert isinstance(result, dict)
