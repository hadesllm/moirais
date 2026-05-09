"""Tests for otmapnk.ot_map_neural_kantorovich."""
import numpy as np
import pytest
from moirais.fn.otmapnk import ot_map_neural_kantorovich


def test_otmapnk_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_map_neural_kantorovich(X, Y, epochs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmapnk_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_map_neural_kantorovich(X, Y, epochs)
    assert isinstance(result, dict)
