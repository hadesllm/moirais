"""Tests for alctxemb.alammar_contextualized_embedding."""
import numpy as np
import pytest
from moirais.fn.alctxemb import alammar_contextualized_embedding


def test_alctxemb_basic():
    """Test basic functionality."""
    layer_outputs = np.random.default_rng(42).normal(0, 1, 100)
    layer_idx = np.random.default_rng(42).normal(0, 1, 100)
    position = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_contextualized_embedding(layer_outputs, layer_idx, position)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alctxemb_edge():
    """Test edge cases."""
    layer_outputs = np.random.default_rng(42).normal(0, 1, 100)
    layer_idx = np.random.default_rng(42).normal(0, 1, 100)
    position = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_contextualized_embedding(layer_outputs, layer_idx, position)
    assert isinstance(result, dict)
