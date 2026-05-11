"""Tests for areN.asymptotic_relative_efficiency."""
import numpy as np
import pytest
from morie.fn.areN import asymptotic_relative_efficiency


def test_areN_basic():
    """Test basic functionality."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = asymptotic_relative_efficiency(estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_areN_edge():
    """Test edge cases."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = asymptotic_relative_efficiency(estimator)
    assert isinstance(result, dict)
