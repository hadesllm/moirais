"""Tests for pseudo.path_specific_effect."""
import numpy as np
import pytest
from morie.fn.pseudo import path_specific_effect


def test_pseudo_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M_list = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = path_specific_effect(Y, X, M_list, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pseudo_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M_list = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = path_specific_effect(Y, X, M_list, path)
    assert isinstance(result, dict)
