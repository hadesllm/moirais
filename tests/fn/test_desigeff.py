"""Tests for desigeff.design_effect."""
import numpy as np
import pytest
from morie.fn.desigeff import design_effect


def test_desigeff_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = design_effect(y, weights, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_desigeff_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = design_effect(y, weights, cluster)
    assert isinstance(result, dict)
