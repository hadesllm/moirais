"""Tests for bndlmm.bound_linear_min_max."""
import numpy as np
import pytest
from moirais.fn.bndlmm import bound_linear_min_max


def test_bndlmm_basic():
    """Test basic functionality."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_linear_min_max(theta, moments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndlmm_edge():
    """Test edge cases."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_linear_min_max(theta, moments)
    assert isinstance(result, dict)
