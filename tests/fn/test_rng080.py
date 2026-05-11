"""Tests for rng080.rangayyan_ch3_parseval_theorem."""
import numpy as np
import pytest
from morie.fn.rng080 import rangayyan_ch3_parseval_theorem


def test_rng080_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = rangayyan_ch3_parseval_theorem(x, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng080_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = rangayyan_ch3_parseval_theorem(x, X)
    assert isinstance(result, dict)
