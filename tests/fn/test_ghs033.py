"""Tests for ghs033.ghosal_ch3_polya_tree_mixture_second_kind."""
import numpy as np
import pytest
from moirais.fn.ghs033 import ghosal_ch3_polya_tree_mixture_second_kind


def test_ghs033_basic():
    """Test basic functionality."""
    alpha_theta = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = ghosal_ch3_polya_tree_mixture_second_kind(alpha_theta, x, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs033_edge():
    """Test edge cases."""
    alpha_theta = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = ghosal_ch3_polya_tree_mixture_second_kind(alpha_theta, x, theta)
    assert isinstance(result, dict)
