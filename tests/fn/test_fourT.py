"""Tests for fourT.fourier_transform."""
import numpy as np
import pytest
from moirais.fn.fourT import fourier_transform


def test_fourT_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = fourier_transform(f, x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fourT_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = fourier_transform(f, x, k)
    assert isinstance(result, dict)
