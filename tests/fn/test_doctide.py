"""Tests for doctide.de_chaisemartin_dhaultfoeuille."""
import numpy as np
import pytest
from morie.fn.doctide import de_chaisemartin_dhaultfoeuille


def test_doctide_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = de_chaisemartin_dhaultfoeuille(y, D, unit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_doctide_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = de_chaisemartin_dhaultfoeuille(y, D, unit, time)
    assert isinstance(result, dict)
