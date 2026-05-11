"""Tests for cnsRos.rosenbaum_bound_signed."""
import numpy as np
import pytest
from morie.fn.cnsRos import rosenbaum_bound_signed


def test_cnsRos_basic():
    """Test basic functionality."""
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    Gamma = np.random.default_rng(42).normal(0, 1, 100)
    result = rosenbaum_bound_signed(pairs, Gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnsRos_edge():
    """Test edge cases."""
    pairs = np.random.default_rng(42).normal(0, 1, 100)
    Gamma = np.random.default_rng(42).normal(0, 1, 100)
    result = rosenbaum_bound_signed(pairs, Gamma)
    assert isinstance(result, dict)
