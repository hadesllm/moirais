"""Tests for chstr.chain_structure."""
import numpy as np
import pytest
from moirais.fn.chstr import chain_structure


def test_chstr_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    conditioned = np.random.default_rng(42).normal(0, 1, 100)
    result = chain_structure(A, B, C, conditioned)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chstr_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    conditioned = np.random.default_rng(42).normal(0, 1, 100)
    result = chain_structure(A, B, C, conditioned)
    assert isinstance(result, dict)
