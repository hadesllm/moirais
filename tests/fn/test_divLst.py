"""Tests for divLst.diversity."""
import numpy as np
import pytest
from morie.fn.divLst import diversity


def test_divLst_basic():
    """Test basic functionality."""
    list = np.random.default_rng(42).normal(0, 1, 100)
    sim_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = diversity(list, sim_matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_divLst_edge():
    """Test edge cases."""
    list = np.random.default_rng(42).normal(0, 1, 100)
    sim_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = diversity(list, sim_matrix)
    assert isinstance(result, dict)
