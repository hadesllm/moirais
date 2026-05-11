"""Tests for rghomo.rangayyan_homomorphic."""
import numpy as np
import pytest
from morie.fn.rghomo import rangayyan_homomorphic


def test_rghomo_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filter_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic(x, filter_type)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghomo_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filter_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_homomorphic(x, filter_type)
    assert isinstance(result, dict)
