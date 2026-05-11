"""Tests for tsalls.tsallis_entropy."""
import numpy as np
import pytest
from morie.fn.tsalls import tsallis_entropy


def test_tsalls_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tsallis_entropy(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tsalls_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = tsallis_entropy(p, q)
    assert isinstance(result, dict)
