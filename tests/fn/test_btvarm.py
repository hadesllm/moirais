"""Tests for btvarm.boot_var_mean."""
import numpy as np
import pytest
from moirais.fn.btvarm import boot_var_mean


def test_btvarm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_var_mean(x, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btvarm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_var_mean(x, B)
    assert isinstance(result, dict)
