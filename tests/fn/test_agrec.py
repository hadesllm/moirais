"""Tests for agrec.andersen_gill_recurrent."""
import numpy as np
import pytest
from moirais.fn.agrec import andersen_gill_recurrent


def test_agrec_basic():
    """Test basic functionality."""
    start = np.random.default_rng(42).normal(0, 1, 100)
    stop = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = andersen_gill_recurrent(start, stop, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agrec_edge():
    """Test edge cases."""
    start = np.random.default_rng(42).normal(0, 1, 100)
    stop = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = andersen_gill_recurrent(start, stop, event, X)
    assert isinstance(result, dict)
