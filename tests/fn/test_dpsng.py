"""Tests for dpsng.dp_singularity_test."""
import numpy as np
import pytest
from morie.fn.dpsng import dp_singularity_test


def test_dpsng_basic():
    """Test basic functionality."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dp_singularity_test(partition, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dpsng_edge():
    """Test edge cases."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dp_singularity_test(partition, alpha)
    assert isinstance(result, dict)
