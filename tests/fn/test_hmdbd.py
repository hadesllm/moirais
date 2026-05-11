"""Tests for hmdbd.geron_decision_boundary."""
import numpy as np
import pytest
from morie.fn.hmdbd import geron_decision_boundary


def test_hmdbd_basic():
    """Test basic functionality."""
    theta = 0.0
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_decision_boundary(theta, X_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdbd_edge():
    """Test edge cases."""
    theta = 0.0
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_decision_boundary(theta, X_grid)
    assert isinstance(result, dict)
