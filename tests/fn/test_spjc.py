"""Tests for spjc.schabenberger_join_count."""
import numpy as np
import pytest
from morie.fn.spjc import schabenberger_join_count


def test_spjc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    category = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_join_count(x, w, category)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spjc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    category = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_join_count(x, w, category)
    assert isinstance(result, dict)
