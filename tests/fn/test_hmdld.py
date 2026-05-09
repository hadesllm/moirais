"""Tests for hmdld.geron_dataloader."""
import numpy as np
import pytest
from moirais.fn.hmdld import geron_dataloader


def test_hmdld_basic():
    """Test basic functionality."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dataloader(dataset, batch_size, shuffle)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdld_edge():
    """Test edge cases."""
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    batch_size = 100
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dataloader(dataset, batch_size, shuffle)
    assert isinstance(result, dict)
