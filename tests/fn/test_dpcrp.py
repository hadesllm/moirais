"""Tests for dpcrp.chinese_restaurant_process."""
import numpy as np
import pytest
from morie.fn.dpcrp import chinese_restaurant_process


def test_dpcrp_basic():
    """Test basic functionality."""
    n = 100
    alpha = 0.05
    result = chinese_restaurant_process(n, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpcrp_edge():
    """Test edge cases."""
    n = 100
    alpha = 0.05
    result = chinese_restaurant_process(n, alpha)
    assert isinstance(result, dict)
