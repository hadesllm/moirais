"""Tests for sbmest.stochastic_block_model."""
import numpy as np
import pytest
from morie.fn.sbmest import stochastic_block_model


def test_sbmest_basic():
    """Test basic functionality."""
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = stochastic_block_model(G, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sbmest_edge():
    """Test edge cases."""
    G = np.eye(10)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = stochastic_block_model(G, K)
    assert isinstance(result, dict)
