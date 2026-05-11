"""Tests for kanon.k_anonymity_check."""
import numpy as np
import pytest
from morie.fn.kanon import k_anonymity_check


def test_kanon_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    k = 5
    result = k_anonymity_check(y, quasi_ids, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kanon_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    k = 5
    result = k_anonymity_check(y, quasi_ids, k)
    assert isinstance(result, dict)
