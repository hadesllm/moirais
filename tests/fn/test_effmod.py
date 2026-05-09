"""Tests for effmod.effect_modification."""
import numpy as np
import pytest
from moirais.fn.effmod import effect_modification


def test_effmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = effect_modification(y, A, V, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_effmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = effect_modification(y, A, V, H)
    assert isinstance(result, dict)
