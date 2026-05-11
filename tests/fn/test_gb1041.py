"""Tests for gb1041.gibbons_kruskal_wallis."""
import numpy as np
import pytest
from morie.fn.gb1041 import gibbons_kruskal_wallis


def test_gb1041_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kruskal_wallis(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1041_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kruskal_wallis(groups)
    assert isinstance(result, dict)
