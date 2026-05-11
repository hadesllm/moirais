"""Tests for hmhplm.geron_hidden_layers_heuristic."""
import numpy as np
import pytest
from morie.fn.hmhplm import geron_hidden_layers_heuristic


def test_hmhplm_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_hidden_layers_heuristic(model, X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhplm_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_hidden_layers_heuristic(model, X, y)
    assert isinstance(result, dict)
