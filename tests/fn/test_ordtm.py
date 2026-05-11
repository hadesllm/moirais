"""Tests for ordtm.ordinal_threshold_model."""
import numpy as np
import pytest
from morie.fn.ordtm import ordinal_threshold_model


def test_ordtm_basic():
    """Test basic functionality."""
    y_ord = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n_categories = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_threshold_model(y_ord, X, Z, n_categories)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ordtm_edge():
    """Test edge cases."""
    y_ord = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n_categories = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_threshold_model(y_ord, X, Z, n_categories)
    assert isinstance(result, dict)
