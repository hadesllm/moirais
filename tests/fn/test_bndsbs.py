"""Tests for bndsbs.bound_subset_inference."""
import numpy as np
import pytest
from moirais.fn.bndsbs import bound_subset_inference


def test_bndsbs_basic():
    """Test basic functionality."""
    theta_full = np.random.default_rng(42).normal(0, 1, 100)
    subset_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_subset_inference(theta_full, subset_idx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndsbs_edge():
    """Test edge cases."""
    theta_full = np.random.default_rng(42).normal(0, 1, 100)
    subset_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_subset_inference(theta_full, subset_idx)
    assert isinstance(result, dict)
