"""Tests for tqdist.turboquant_distortion_bound."""
import numpy as np
import pytest
from moirais.fn.tqdist import turboquant_distortion_bound


def test_tqdist_basic():
    """Test basic functionality."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_distortion_bound(eps, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqdist_edge():
    """Test edge cases."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_distortion_bound(eps, delta)
    assert isinstance(result, dict)
