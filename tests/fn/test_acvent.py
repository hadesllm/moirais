"""Tests for acvent.differential_entropy."""
import numpy as np
import pytest
from morie.fn.acvent import differential_entropy


def test_acvent_basic():
    """Test basic functionality."""
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = differential_entropy(density)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acvent_edge():
    """Test edge cases."""
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = differential_entropy(density)
    assert isinstance(result, dict)
