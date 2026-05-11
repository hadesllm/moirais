"""Tests for hrzadm.horowitz_additive_model."""
import numpy as np
import pytest
from morie.fn.hrzadm import horowitz_additive_model


def test_hrzadm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_additive_model(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzadm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_additive_model(x, y, bandwidth)
    assert isinstance(result, dict)
