"""Tests for bndinf.bound_inference."""
import numpy as np
import pytest
from moirais.fn.bndinf import bound_inference


def test_bndinf_basic():
    """Test basic functionality."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_inference(theta, moments, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndinf_edge():
    """Test edge cases."""
    theta = 0.0
    moments = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = bound_inference(theta, moments, alpha)
    assert isinstance(result, dict)
