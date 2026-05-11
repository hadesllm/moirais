"""Tests for irtras.rating_scale_model."""
import numpy as np
import pytest
from morie.fn.irtras import rating_scale_model


def test_irtras_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = rating_scale_model(X, ncats)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtras_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = rating_scale_model(X, ncats)
    assert isinstance(result, dict)
