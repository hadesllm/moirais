"""Tests for hmagrd.geron_autograd."""
import numpy as np
import pytest
from moirais.fn.hmagrd import geron_autograd


def test_hmagrd_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = geron_autograd(loss, params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmagrd_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = geron_autograd(loss, params)
    assert isinstance(result, dict)
