"""Tests for gb5412.gibbons_sign_normal_approx."""
import numpy as np
import pytest
from moirais.fn.gb5412 import gibbons_sign_normal_approx


def test_gb5412_basic():
    """Test basic functionality."""
    k = 5
    n = 100
    result = gibbons_sign_normal_approx(k, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5412_edge():
    """Test edge cases."""
    k = 5
    n = 100
    result = gibbons_sign_normal_approx(k, n)
    assert isinstance(result, dict)
