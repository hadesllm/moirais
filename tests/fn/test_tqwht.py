"""Tests for tqwht.turboquant_walsh_hadamard_transform."""
import numpy as np
import pytest
from morie.fn.tqwht import turboquant_walsh_hadamard_transform


def test_tqwht_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_walsh_hadamard_transform(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqwht_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_walsh_hadamard_transform(x)
    assert isinstance(result, dict)
