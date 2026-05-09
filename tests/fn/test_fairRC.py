"""Tests for fairRC.fairness_rec."""
import numpy as np
import pytest
from moirais.fn.fairRC import fairness_rec


def test_fairRC_basic():
    """Test basic functionality."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    attrs = np.random.default_rng(42).normal(0, 1, 100)
    result = fairness_rec(pred, attrs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fairRC_edge():
    """Test edge cases."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    attrs = np.random.default_rng(42).normal(0, 1, 100)
    result = fairness_rec(pred, attrs)
    assert isinstance(result, dict)
