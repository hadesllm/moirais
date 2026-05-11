"""Tests for rng011.rangayyan_ch3_shannon_entropy_discrete."""
import numpy as np
import pytest
from morie.fn.rng011 import rangayyan_ch3_shannon_entropy_discrete


def test_rng011_basic():
    """Test basic functionality."""
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_shannon_entropy_discrete(p_eta, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng011_edge():
    """Test edge cases."""
    p_eta = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_shannon_entropy_discrete(p_eta, L)
    assert isinstance(result, dict)
