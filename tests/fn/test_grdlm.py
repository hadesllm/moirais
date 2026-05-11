"""Tests for grdlm.geron_dataloader_minibatch."""
import numpy as np
import pytest
from morie.fn.grdlm import geron_dataloader_minibatch


def test_grdlm_basic():
    """Test basic functionality."""
    n = 100
    b = np.random.default_rng(42).normal(0, 1, 100)
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_dataloader_minibatch(n, b, shuffle, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdlm_edge():
    """Test edge cases."""
    n = 100
    b = np.random.default_rng(42).normal(0, 1, 100)
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_dataloader_minibatch(n, b, shuffle, seed)
    assert isinstance(result, dict)
