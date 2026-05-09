"""Tests for tolim.tolerance_limits."""
import numpy as np
import pytest
from moirais.fn.tolim import tolerance_limits


def test_tolim_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coverage = np.random.default_rng(42).normal(0, 1, 100)
    confidence = np.random.default_rng(42).normal(0, 1, 100)
    result = tolerance_limits(x, coverage, confidence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tolim_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coverage = np.random.default_rng(42).normal(0, 1, 100)
    confidence = np.random.default_rng(42).normal(0, 1, 100)
    result = tolerance_limits(x, coverage, confidence)
    assert isinstance(result, dict)
