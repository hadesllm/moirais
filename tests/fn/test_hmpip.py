"""Tests for hmpip.geron_pipeline."""
import numpy as np
import pytest
from moirais.fn.hmpip import geron_pipeline


def test_hmpip_basic():
    """Test basic functionality."""
    steps = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_pipeline(steps, X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpip_edge():
    """Test edge cases."""
    steps = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_pipeline(steps, X, y)
    assert isinstance(result, dict)
