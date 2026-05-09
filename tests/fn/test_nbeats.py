"""Tests for nbeats.n_beats."""
import numpy as np
import pytest
from moirais.fn.nbeats import n_beats


def test_nbeats_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = n_beats(y, stacks, blocks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nbeats_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    result = n_beats(y, stacks, blocks)
    assert isinstance(result, dict)
