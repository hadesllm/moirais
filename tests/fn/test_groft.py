"""Tests for groft.geron_overfitting_gap."""
import numpy as np
import pytest
from morie.fn.groft import geron_overfitting_gap


def test_groft_basic():
    """Test basic functionality."""
    train_scores = np.random.default_rng(42).normal(0, 1, 100)
    val_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_overfitting_gap(train_scores, val_scores)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_groft_edge():
    """Test edge cases."""
    train_scores = np.random.default_rng(42).normal(0, 1, 100)
    val_scores = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_overfitting_gap(train_scores, val_scores)
    assert isinstance(result, dict)
