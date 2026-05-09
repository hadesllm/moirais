"""Tests for ngnest.n_beats."""
import numpy as np
import pytest
from moirais.fn.ngnest import n_beats


def test_ngnest_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    result = n_beats(y, horizon, stacks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ngnest_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    result = n_beats(y, horizon, stacks)
    assert isinstance(result, dict)
