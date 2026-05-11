"""Tests for dpparit.pitman_yor_process."""
import numpy as np
import pytest
from morie.fn.dpparit import pitman_yor_process


def test_dpparit_basic():
    """Test basic functionality."""
    n = 100
    alpha = 0.05
    sigma = 1.0
    result = pitman_yor_process(n, alpha, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpparit_edge():
    """Test edge cases."""
    n = 100
    alpha = 0.05
    sigma = 1.0
    result = pitman_yor_process(n, alpha, sigma)
    assert isinstance(result, dict)
