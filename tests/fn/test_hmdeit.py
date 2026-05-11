"""Tests for hmdeit.geron_deit."""
import numpy as np
import pytest
from morie.fn.hmdeit import geron_deit


def test_hmdeit_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deit(image, patch_size, n_layers, teacher)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdeit_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    n_layers = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_deit(image, patch_size, n_layers, teacher)
    assert isinstance(result, dict)
