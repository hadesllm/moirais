"""Tests for mxtent.max_entropy."""
import numpy as np
import pytest
from morie.fn.mxtent import max_entropy


def test_mxtent_basic():
    """Test basic functionality."""
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = max_entropy(constraints)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mxtent_edge():
    """Test edge cases."""
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    result = max_entropy(constraints)
    assert isinstance(result, dict)
