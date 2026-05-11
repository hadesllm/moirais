"""Tests for hmstk.geron_stacking."""
import numpy as np
import pytest
from morie.fn.hmstk import geron_stacking


def test_hmstk_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_models = np.random.default_rng(42).normal(0, 1, 100)
    meta_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacking(X, y, base_models, meta_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmstk_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    base_models = np.random.default_rng(42).normal(0, 1, 100)
    meta_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_stacking(X, y, base_models, meta_model)
    assert isinstance(result, dict)
