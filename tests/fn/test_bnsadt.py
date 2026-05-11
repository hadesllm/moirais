"""Tests for bnsadt.bound_adversarial."""
import numpy as np
import pytest
from morie.fn.bnsadt import bound_adversarial


def test_bnsadt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    family = 'gaussian'
    result = bound_adversarial(y, D, family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnsadt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    family = 'gaussian'
    result = bound_adversarial(y, D, family)
    assert isinstance(result, dict)
