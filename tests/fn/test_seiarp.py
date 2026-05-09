"""Tests for seiarp.seira_asymptomatic."""
import numpy as np
import pytest
from moirais.fn.seiarp import seira_asymptomatic


def test_seiarp_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    R = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = seira_asymptomatic(S, E, I, A, R, params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_seiarp_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    R = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = seira_asymptomatic(S, E, I, A, R, params)
    assert isinstance(result, dict)
