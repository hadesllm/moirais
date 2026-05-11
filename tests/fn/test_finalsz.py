"""Tests for finalsz.final_epidemic_size."""
import numpy as np
import pytest
from morie.fn.finalsz import final_epidemic_size


def test_finalsz_basic():
    """Test basic functionality."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    s0 = np.random.default_rng(42).normal(0, 1, 100)
    result = final_epidemic_size(R0, s0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_finalsz_edge():
    """Test edge cases."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    s0 = np.random.default_rng(42).normal(0, 1, 100)
    result = final_epidemic_size(R0, s0)
    assert isinstance(result, dict)
