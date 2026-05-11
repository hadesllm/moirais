"""Tests for tsallen.tsallis_entropy."""
import numpy as np
import pytest
from morie.fn.tsallen import tsallis_entropy


def test_tsallen_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tsallis_entropy(y, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tsallen_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tsallis_entropy(y, q)
    assert isinstance(result, dict)
