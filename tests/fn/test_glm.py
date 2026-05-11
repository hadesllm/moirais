"""Tests for glm.glr_test."""
import numpy as np
import pytest
from morie.fn.glm import glr_test


def test_glm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p0 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    result = glr_test(x, p0, p1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_glm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p0 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    result = glr_test(x, p0, p1)
    assert isinstance(result, dict)
