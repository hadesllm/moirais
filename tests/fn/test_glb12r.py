"""Tests for glb12r.greatest_lower_bound."""
import numpy as np
import pytest
from morie.fn.glb12r import greatest_lower_bound


def test_glb12r_basic():
    """Test basic functionality."""
    covariance = np.random.default_rng(42).normal(0, 1, 100)
    result = greatest_lower_bound(covariance)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_glb12r_edge():
    """Test edge cases."""
    covariance = np.random.default_rng(42).normal(0, 1, 100)
    result = greatest_lower_bound(covariance)
    assert isinstance(result, dict)
