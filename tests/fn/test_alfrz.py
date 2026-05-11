"""Tests for alfrz.alammar_layer_freezing."""
import numpy as np
import pytest
from morie.fn.alfrz import alammar_layer_freezing


def test_alfrz_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_layer_freezing(model, schedule)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfrz_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_layer_freezing(model, schedule)
    assert isinstance(result, dict)
