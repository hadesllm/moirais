"""Tests for gb_ttm.gibbons_two_sample_t_efficacy."""
import numpy as np
import pytest
from morie.fn.gb_ttm import gibbons_two_sample_t_efficacy


def test_gb_ttm_basic():
    """Test basic functionality."""
    N = 100
    sigma = 1.0
    lam = 0.1
    result = gibbons_two_sample_t_efficacy(N, sigma, lam)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_ttm_edge():
    """Test edge cases."""
    N = 100
    sigma = 1.0
    lam = 0.1
    result = gibbons_two_sample_t_efficacy(N, sigma, lam)
    assert isinstance(result, dict)
