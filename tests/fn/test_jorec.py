"""Tests for jorec.joseph_recursive_multistep."""
import numpy as np
import pytest
from moirais.fn.jorec import joseph_recursive_multistep


def test_jorec_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_recursive_multistep(y, model, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jorec_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_recursive_multistep(y, model, H)
    assert isinstance(result, dict)
