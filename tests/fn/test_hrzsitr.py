"""Tests for hrzsitr.horowitz_sieve_npiv."""
import numpy as np
import pytest
from moirais.fn.hrzsitr import horowitz_sieve_npiv


def test_hrzsitr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = horowitz_sieve_npiv(x, y, w, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsitr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = horowitz_sieve_npiv(x, y, w, K)
    assert isinstance(result, dict)
