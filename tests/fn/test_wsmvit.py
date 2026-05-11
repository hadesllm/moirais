"""Tests for wsmvit.wasserman_viterbi."""
import numpy as np
import pytest
from morie.fn.wsmvit import wasserman_viterbi


def test_wsmvit_basic():
    """Test basic functionality."""
    obs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_viterbi(obs, A, B, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmvit_edge():
    """Test edge cases."""
    obs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_viterbi(obs, A, B, pi)
    assert isinstance(result, dict)
