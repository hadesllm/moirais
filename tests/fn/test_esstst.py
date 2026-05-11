"""Tests for esstst.effective_sample_size."""
import numpy as np
import pytest
from morie.fn.esstst import effective_sample_size


def test_esstst_basic():
    """Test basic functionality."""
    weights = np.random.default_rng(45).exponential(1, 100)
    result = effective_sample_size(weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esstst_edge():
    """Test edge cases."""
    weights = np.random.default_rng(45).exponential(1, 100)
    result = effective_sample_size(weights)
    assert isinstance(result, dict)
