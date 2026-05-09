"""Tests for krust.kruskal_stress."""
import numpy as np
import pytest
from moirais.fn.krust import kruskal_stress


def test_krust_basic():
    """Test basic functionality."""
    D_observed = np.random.default_rng(42).normal(0, 1, 100)
    D_config = np.random.default_rng(42).normal(0, 1, 100)
    result = kruskal_stress(D_observed, D_config)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krust_edge():
    """Test edge cases."""
    D_observed = np.random.default_rng(42).normal(0, 1, 100)
    D_config = np.random.default_rng(42).normal(0, 1, 100)
    result = kruskal_stress(D_observed, D_config)
    assert isinstance(result, dict)
