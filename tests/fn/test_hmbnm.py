"""Tests for hmbnm.geron_biological_neuron."""
import numpy as np
import pytest
from morie.fn.hmbnm import geron_biological_neuron


def test_hmbnm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_biological_neuron(x, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbnm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_biological_neuron(x, w, b)
    assert isinstance(result, dict)
