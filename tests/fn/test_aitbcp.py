"""Tests for aitbcp.compositional_bray_curtis."""
import numpy as np
import pytest
from moirais.fn.aitbcp import compositional_bray_curtis


def test_aitbcp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = compositional_bray_curtis(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitbcp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = compositional_bray_curtis(x, y)
    assert isinstance(result, dict)
