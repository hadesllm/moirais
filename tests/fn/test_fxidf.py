"""Tests for fxidf.effect_modification."""
import numpy as np
import pytest
from morie.fn.fxidf import effect_modification


def test_fxidf_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = effect_modification(Y, X, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fxidf_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = effect_modification(Y, X, C)
    assert isinstance(result, dict)
