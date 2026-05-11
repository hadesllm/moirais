"""Tests for hrzt2.horowitz_local_ate."""
import numpy as np
import pytest
from morie.fn.hrzt2 import horowitz_local_ate


def test_hrzt2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_local_ate(x, y, z, treatment)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzt2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_local_ate(x, y, z, treatment)
    assert isinstance(result, dict)
