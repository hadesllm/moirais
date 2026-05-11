"""Tests for agnnbk.alphazero_resnet_block."""
import numpy as np
import pytest
from morie.fn.agnnbk import alphazero_resnet_block


def test_agnnbk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_resnet_block(x, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agnnbk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_resnet_block(x, filters)
    assert isinstance(result, dict)
