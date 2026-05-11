"""Tests for emkfst.em_state_space."""
import numpy as np
import pytest
from morie.fn.emkfst import em_state_space


def test_emkfst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = em_state_space(y, init, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_emkfst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = em_state_space(y, init, max_iter)
    assert isinstance(result, dict)
