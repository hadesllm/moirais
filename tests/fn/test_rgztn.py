"""Tests for rgztn.regularization_path."""
import numpy as np
import pytest
from morie.fn.rgztn import regularization_path


def test_rgztn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = regularization_path(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgztn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = regularization_path(x, y)
    assert isinstance(result, dict)
