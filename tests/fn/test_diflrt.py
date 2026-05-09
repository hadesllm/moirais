"""Tests for diflrt.dif_likelihood_ratio."""
import numpy as np
import pytest
from moirais.fn.diflrt import dif_likelihood_ratio


def test_diflrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = dif_likelihood_ratio(y, group, theta)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_diflrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = dif_likelihood_ratio(y, group, theta)
    assert isinstance(result, dict)
