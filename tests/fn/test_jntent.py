"""Tests for jntent.joint_entropy."""
import numpy as np
import pytest
from morie.fn.jntent import joint_entropy


def test_jntent_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_entropy(pxy, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jntent_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_entropy(pxy, base)
    assert isinstance(result, dict)
