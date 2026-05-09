"""Tests for hybRC.hybrid_rec."""
import numpy as np
import pytest
from moirais.fn.hybRC import hybrid_rec


def test_hybRC_basic():
    """Test basic functionality."""
    scores_cf = np.random.default_rng(42).normal(0, 1, 100)
    scores_cb = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = hybrid_rec(scores_cf, scores_cb, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hybRC_edge():
    """Test edge cases."""
    scores_cf = np.random.default_rng(42).normal(0, 1, 100)
    scores_cb = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = hybrid_rec(scores_cf, scores_cb, alpha)
    assert isinstance(result, dict)
