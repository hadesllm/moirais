"""Tests for effnnt.efficientnet_block."""
import numpy as np
import pytest
from moirais.fn.effnnt import efficientnet_block


def test_effnnt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    expand_ratio = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = efficientnet_block(x, expand_ratio, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_effnnt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    expand_ratio = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = efficientnet_block(x, expand_ratio, filters)
    assert isinstance(result, dict)
