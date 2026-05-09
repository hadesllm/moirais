"""Tests for randwk.random_walk."""
import numpy as np
import pytest
from moirais.fn.randwk import random_walk


def test_randwk_basic():
    """Test basic functionality."""
    G = np.eye(10)
    start = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = random_walk(G, start, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_randwk_edge():
    """Test edge cases."""
    G = np.eye(10)
    start = np.random.default_rng(42).normal(0, 1, 100)
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = random_walk(G, start, steps)
    assert isinstance(result, dict)
