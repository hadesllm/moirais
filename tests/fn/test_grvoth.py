"""Tests for grvoth.geron_hard_voting."""
import numpy as np
import pytest
from moirais.fn.grvoth import geron_hard_voting


def test_grvoth_basic():
    """Test basic functionality."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grvoth_edge():
    """Test edge cases."""
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hard_voting(predictions)
    assert isinstance(result, dict)
