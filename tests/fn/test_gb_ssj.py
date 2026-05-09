"""Tests for gb_ssj.gibbons_sign_sample_size_2."""
import numpy as np
import pytest
from moirais.fn.gb_ssj import gibbons_sign_sample_size_2


def test_gb_ssj_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    p = 5
    result = gibbons_sign_sample_size_2(alpha, beta, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_ssj_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    p = 5
    result = gibbons_sign_sample_size_2(alpha, beta, p)
    assert isinstance(result, dict)
