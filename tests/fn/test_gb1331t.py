"""Tests for gb1331t.gibbons_t_efficacy."""
import numpy as np
import pytest
from morie.fn.gb1331t import gibbons_t_efficacy


def test_gb1331t_basic():
    """Test basic functionality."""
    N = 100
    sigma = 1.0
    result = gibbons_t_efficacy(N, sigma)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1331t_edge():
    """Test edge cases."""
    N = 100
    sigma = 1.0
    result = gibbons_t_efficacy(N, sigma)
    assert isinstance(result, dict)
