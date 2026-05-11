"""Tests for morrisM.morris_screening."""
import numpy as np
import pytest
from morie.fn.morrisM import morris_screening


def test_morrisM_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = morris_screening(model, input_dist, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_morrisM_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = morris_screening(model, input_dist, N)
    assert isinstance(result, dict)
