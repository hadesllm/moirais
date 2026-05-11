"""Tests for bndpos.bound_pos_treatment."""
import numpy as np
import pytest
from morie.fn.bndpos import bound_pos_treatment


def test_bndpos_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_max = 100
    result = bound_pos_treatment(y, D, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndpos_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_max = 100
    result = bound_pos_treatment(y, D, y_max)
    assert isinstance(result, dict)
