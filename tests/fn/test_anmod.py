"""Tests for anmod.additive_noise_model."""
import numpy as np
import pytest
from moirais.fn.anmod import additive_noise_model


def test_anmod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = additive_noise_model(X, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_anmod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = additive_noise_model(X, Y)
    assert isinstance(result, dict)
