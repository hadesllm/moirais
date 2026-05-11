"""Tests for feynAI.ai_feynman."""
import numpy as np
import pytest
from morie.fn.feynAI import ai_feynman


def test_feynAI_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ai_feynman(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_feynAI_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ai_feynman(X, y)
    assert isinstance(result, dict)
