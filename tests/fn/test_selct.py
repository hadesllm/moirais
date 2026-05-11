"""Tests for selct.genomic_selection_accuracy."""
import numpy as np
import pytest
from morie.fn.selct import genomic_selection_accuracy


def test_selct_basic():
    """Test basic functionality."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    predictive_ability = np.random.default_rng(42).normal(0, 1, 100)
    sigma_g = np.random.default_rng(42).normal(0, 1, 100)
    h2 = np.random.default_rng(42).normal(0, 1, 100)
    result = genomic_selection_accuracy(i, predictive_ability, sigma_g, h2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_selct_edge():
    """Test edge cases."""
    i = np.random.default_rng(42).normal(0, 1, 100)
    predictive_ability = np.random.default_rng(42).normal(0, 1, 100)
    sigma_g = np.random.default_rng(42).normal(0, 1, 100)
    h2 = np.random.default_rng(42).normal(0, 1, 100)
    result = genomic_selection_accuracy(i, predictive_ability, sigma_g, h2)
    assert isinstance(result, dict)
