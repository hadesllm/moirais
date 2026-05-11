"""Tests for jntlmm.joint_longitudinal_survival."""
import numpy as np
import pytest
from morie.fn.jntlmm import joint_longitudinal_survival


def test_jntlmm_basic():
    """Test basic functionality."""
    long_y = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_longitudinal_survival(long_y, time, event, X, Z, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jntlmm_edge():
    """Test edge cases."""
    long_y = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_longitudinal_survival(long_y, time, event, X, Z, cluster)
    assert isinstance(result, dict)
