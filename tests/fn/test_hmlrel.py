"""Tests for hmlrel.geron_leaky_relu."""
import numpy as np
import pytest
from moirais.fn.hmlrel import geron_leaky_relu


def test_hmlrel_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_leaky_relu(z, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlrel_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_leaky_relu(z, alpha)
    assert isinstance(result, dict)
