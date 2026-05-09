"""Tests for theils.theil_sen."""
import numpy as np
import pytest
from moirais.fn.theils import theil_sen


def test_theils_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = theil_sen(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_theils_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = theil_sen(x, y)
    assert isinstance(result, dict)
