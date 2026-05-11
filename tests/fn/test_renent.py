"""Tests for renent.renyi_entropy."""
import numpy as np
import pytest
from morie.fn.renent import renyi_entropy


def test_renent_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = renyi_entropy(y, alpha, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_renent_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = renyi_entropy(y, alpha, base)
    assert isinstance(result, dict)
