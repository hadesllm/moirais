"""Tests for diripr.dirichlet_multinomial."""
import numpy as np
import pytest
from morie.fn.diripr import dirichlet_multinomial


def test_diripr_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dirichlet_multinomial(counts, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diripr_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dirichlet_multinomial(counts, alpha)
    assert isinstance(result, dict)
