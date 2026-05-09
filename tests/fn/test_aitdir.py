"""Tests for aitdir.dirichlet_density."""
import numpy as np
import pytest
from moirais.fn.aitdir import dirichlet_density


def test_aitdir_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dirichlet_density(x, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitdir_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dirichlet_density(x, alpha)
    assert isinstance(result, dict)
