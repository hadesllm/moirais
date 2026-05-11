"""Tests for grtlu.geron_threshold_logic_unit."""
import numpy as np
import pytest
from morie.fn.grtlu import geron_threshold_logic_unit


def test_grtlu_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_threshold_logic_unit(x, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtlu_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_threshold_logic_unit(x, w, b)
    assert isinstance(result, dict)
