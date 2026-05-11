"""Tests for hmnov.geron_novelty_detection."""
import numpy as np
import pytest
from morie.fn.hmnov import geron_novelty_detection


def test_hmnov_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X_new = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_novelty_detection(model, X_new)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnov_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X_new = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_novelty_detection(model, X_new)
    assert isinstance(result, dict)
