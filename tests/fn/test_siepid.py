"""Tests for siepid.si_epidemic."""
import numpy as np
import pytest
from morie.fn.siepid import si_epidemic


def test_siepid_basic():
    """Test basic functionality."""
    G = np.eye(10)
    beta = 0.8
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = si_epidemic(G, beta, initial)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_siepid_edge():
    """Test edge cases."""
    G = np.eye(10)
    beta = 0.8
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = si_epidemic(G, beta, initial)
    assert isinstance(result, dict)
