"""Tests for rfkrn.random_fourier_features."""
import numpy as np
import pytest
from morie.fn.rfkrn import random_fourier_features


def test_rfkrn_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    D = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = random_fourier_features(X, D, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfkrn_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    D = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = random_fourier_features(X, D, kernel)
    assert isinstance(result, dict)
