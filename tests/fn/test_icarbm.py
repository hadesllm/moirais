"""Tests for icarbm.icar_prior."""
import numpy as np
import pytest
from morie.fn.icarbm import icar_prior


def test_icarbm_basic():
    """Test basic functionality."""
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    tau = 0.1
    result = icar_prior(adjacency, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icarbm_edge():
    """Test edge cases."""
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    tau = 0.1
    result = icar_prior(adjacency, tau)
    assert isinstance(result, dict)
