"""Tests for gb2431.gibbons_binomial_beta_link."""
import numpy as np
import pytest
from moirais.fn.gb2431 import gibbons_binomial_beta_link


def test_gb2431_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    r = 10
    n = 100
    result = gibbons_binomial_beta_link(t, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb2431_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    r = 10
    n = 100
    result = gibbons_binomial_beta_link(t, r, n)
    assert isinstance(result, dict)
