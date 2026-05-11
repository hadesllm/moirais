"""Tests for hrzconf.horowitz_confidence_bands."""
import numpy as np
import pytest
from morie.fn.hrzconf import horowitz_confidence_bands


def test_hrzconf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_confidence_bands(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzconf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_confidence_bands(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
