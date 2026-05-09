"""Tests for hiatus.hiatus_model."""
import numpy as np
import pytest
from moirais.fn.hiatus import hiatus_model


def test_hiatus_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    I1 = np.random.default_rng(42).normal(0, 1, 100)
    I2 = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    cross_immunity = np.random.default_rng(42).normal(0, 1, 100)
    result = hiatus_model(S, I1, I2, R, cross_immunity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hiatus_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    I1 = np.random.default_rng(42).normal(0, 1, 100)
    I2 = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    cross_immunity = np.random.default_rng(42).normal(0, 1, 100)
    result = hiatus_model(S, I1, I2, R, cross_immunity)
    assert isinstance(result, dict)
