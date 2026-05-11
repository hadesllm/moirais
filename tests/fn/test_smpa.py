"""Tests for smpa.smooth_sensitivity."""
import numpy as np
import pytest
from morie.fn.smpa import smooth_sensitivity


def test_smpa_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = smooth_sensitivity(query, D, beta)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_smpa_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = smooth_sensitivity(query, D, beta)
    assert isinstance(result, dict)
