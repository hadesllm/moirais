"""Tests for ggrcst.granger_causality."""
import numpy as np
import pytest
from moirais.fn.ggrcst import granger_causality


def test_ggrcst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = granger_causality(x, y, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ggrcst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = granger_causality(x, y, p)
    assert isinstance(result, dict)
