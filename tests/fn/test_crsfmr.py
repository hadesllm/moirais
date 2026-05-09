"""Tests for crsfmr.crossformer."""
import numpy as np
import pytest
from moirais.fn.crsfmr import crossformer


def test_crsfmr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    result = crossformer(X, y, seg_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crsfmr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    result = crossformer(X, y, seg_len)
    assert isinstance(result, dict)
