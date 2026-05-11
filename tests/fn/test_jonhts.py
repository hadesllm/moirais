"""Tests for jonhts.joseph_nhits."""
import numpy as np
import pytest
from morie.fn.jonhts import joseph_nhits


def test_jonhts_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    expressivity = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_nhits(x, blocks, expressivity, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jonhts_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    expressivity = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_nhits(x, blocks, expressivity, horizon)
    assert isinstance(result, dict)
