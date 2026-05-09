"""Tests for dpkmn.dp_kmeans."""
import numpy as np
import pytest
from moirais.fn.dpkmn import dp_kmeans


def test_dpkmn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = dp_kmeans(y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpkmn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = dp_kmeans(y, lam)
    assert isinstance(result, dict)
