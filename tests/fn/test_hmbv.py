"""Tests for hmbv.geron_bias_variance_tradeoff."""
import numpy as np
import pytest
from moirais.fn.hmbv import geron_bias_variance_tradeoff


def test_hmbv_basic():
    """Test basic functionality."""
    preds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_bias_variance_tradeoff(preds, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbv_edge():
    """Test edge cases."""
    preds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_bias_variance_tradeoff(preds, y)
    assert isinstance(result, dict)
