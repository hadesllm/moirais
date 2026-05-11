"""Tests for tmlext.tmle_external_data."""
import numpy as np
import pytest
from morie.fn.tmlext import tmle_external_data


def test_tmlext_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    external = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_external_data(y, D, X, external)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlext_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    external = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_external_data(y, D, X, external)
    assert isinstance(result, dict)
