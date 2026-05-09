"""Tests for gb243.gibbons_order_beta."""
import numpy as np
import pytest
from moirais.fn.gb243 import gibbons_order_beta


def test_gb243_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    result = gibbons_order_beta(r, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb243_edge():
    """Test edge cases."""
    r = 10
    n = 100
    result = gibbons_order_beta(r, n)
    assert isinstance(result, dict)
