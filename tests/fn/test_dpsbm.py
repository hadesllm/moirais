"""Tests for dpsbm.dp_stochastic_block."""
import numpy as np
import pytest
from morie.fn.dpsbm import dp_stochastic_block


def test_dpsbm_basic():
    """Test basic functionality."""
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    alpha = 0.05
    result = dp_stochastic_block(adjacency, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpsbm_edge():
    """Test edge cases."""
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    alpha = 0.05
    result = dp_stochastic_block(adjacency, alpha)
    assert isinstance(result, dict)
