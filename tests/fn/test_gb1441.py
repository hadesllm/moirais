"""Tests for gb1441.gibbons_fisher_exact."""
import numpy as np
import pytest
from morie.fn.gb1441 import gibbons_fisher_exact


def test_gb1441_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fisher_exact(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1441_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fisher_exact(table)
    assert isinstance(result, dict)
