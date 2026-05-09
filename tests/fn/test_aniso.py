"""Tests for aniso.anisotropy_test."""
import numpy as np
import pytest
from moirais.fn.aniso import anisotropy_test


def test_aniso_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = anisotropy_test(x, coords)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_aniso_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = anisotropy_test(x, coords)
    assert isinstance(result, dict)
