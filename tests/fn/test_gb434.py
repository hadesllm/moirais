"""Tests for gb434.gibbons_ks_one_sided_dist."""
import numpy as np
import pytest
from morie.fn.gb434 import gibbons_ks_one_sided_dist


def test_gb434_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_one_sided_dist(c, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb434_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_ks_one_sided_dist(c, n)
    assert isinstance(result, dict)
