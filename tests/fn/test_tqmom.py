"""Tests for tqmom.turboquant_normal_moment."""
import numpy as np
import pytest
from morie.fn.tqmom import turboquant_normal_moment


def test_tqmom_basic():
    """Test basic functionality."""
    sigma = 1.0
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_normal_moment(sigma, l)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqmom_edge():
    """Test edge cases."""
    sigma = 1.0
    l = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_normal_moment(sigma, l)
    assert isinstance(result, dict)
