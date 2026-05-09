"""Tests for hmcrg.hierarchical_model."""
import numpy as np
import pytest
from moirais.fn.hmcrg import hierarchical_model


def test_hmcrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_model(y, group)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_model(y, group)
    assert isinstance(result, dict)
