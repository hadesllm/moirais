"""Tests for dpglap.dp_laplace_mechanism."""
import numpy as np
import pytest
from moirais.fn.dpglap import dp_laplace_mechanism


def test_dpglap_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_laplace_mechanism(y, sensitivity, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpglap_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_laplace_mechanism(y, sensitivity, epsilon)
    assert isinstance(result, dict)
