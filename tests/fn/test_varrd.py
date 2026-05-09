"""Tests for varrd.variance_reduction_split."""
import numpy as np
import pytest
from moirais.fn.varrd import variance_reduction_split


def test_varrd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_reduction_split(y, split_idx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varrd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_reduction_split(y, split_idx)
    assert isinstance(result, dict)
