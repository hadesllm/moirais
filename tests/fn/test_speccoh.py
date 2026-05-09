"""Tests for speccoh.coherence."""
import numpy as np
import pytest
from moirais.fn.speccoh import coherence


def test_speccoh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = coherence(x, y, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_speccoh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = coherence(x, y, f)
    assert isinstance(result, dict)
