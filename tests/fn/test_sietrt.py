"""Tests for sietrt.sis_epidemic."""
import numpy as np
import pytest
from morie.fn.sietrt import sis_epidemic


def test_sietrt_basic():
    """Test basic functionality."""
    G = np.eye(10)
    beta = 0.8
    gamma = 1.0
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = sis_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sietrt_edge():
    """Test edge cases."""
    G = np.eye(10)
    beta = 0.8
    gamma = 1.0
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = sis_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
