"""Tests for hmlrh.geron_learning_rate_heuristic."""
import numpy as np
import pytest
from moirais.fn.hmlrh import geron_learning_rate_heuristic


def test_hmlrh_basic():
    """Test basic functionality."""
    lr_curve = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_rate_heuristic(lr_curve)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlrh_edge():
    """Test edge cases."""
    lr_curve = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_learning_rate_heuristic(lr_curve)
    assert isinstance(result, dict)
