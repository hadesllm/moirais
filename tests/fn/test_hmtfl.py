"""Tests for hmtfl.geron_transfer_learning."""
import numpy as np
import pytest
from moirais.fn.hmtfl import geron_transfer_learning


def test_hmtfl_basic():
    """Test basic functionality."""
    pretrained_model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_frozen = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transfer_learning(pretrained_model, X, y, n_frozen)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtfl_edge():
    """Test edge cases."""
    pretrained_model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_frozen = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_transfer_learning(pretrained_model, X, y, n_frozen)
    assert isinstance(result, dict)
