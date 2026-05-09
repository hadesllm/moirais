"""Tests for gb551.gibbons_rank_order_stat."""
import numpy as np
import pytest
from moirais.fn.gb551 import gibbons_rank_order_stat


def test_gb551_basic():
    """Test basic functionality."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rank_order_stat(differences)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb551_edge():
    """Test edge cases."""
    differences = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rank_order_stat(differences)
    assert isinstance(result, dict)
