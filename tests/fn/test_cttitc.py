"""Tests for cttitc.ctt_item_total_corr."""
import numpy as np
import pytest
from moirais.fn.cttitc import ctt_item_total_corr


def test_cttitc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    item_index = np.random.default_rng(42).normal(0, 1, 100)
    result = ctt_item_total_corr(X, item_index)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_cttitc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    item_index = np.random.default_rng(42).normal(0, 1, 100)
    result = ctt_item_total_corr(X, item_index)
    assert isinstance(result, dict)
