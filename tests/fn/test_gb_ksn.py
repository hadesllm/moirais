"""Tests for gb_ksn.gibbons_ks_sample_size."""
import numpy as np
import pytest
from morie.fn.gb_ksn import gibbons_ks_sample_size


def test_gb_ksn_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    alpha = 0.05
    result = gibbons_ks_sample_size(epsilon, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_ksn_edge():
    """Test edge cases."""
    epsilon = 1e-6
    alpha = 0.05
    result = gibbons_ks_sample_size(epsilon, alpha)
    assert isinstance(result, dict)
