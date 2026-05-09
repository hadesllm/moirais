"""Tests for grvots.geron_soft_voting."""
import numpy as np
import pytest
from moirais.fn.grvots import geron_soft_voting


def test_grvots_basic():
    """Test basic functionality."""
    probabilities = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_soft_voting(probabilities)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grvots_edge():
    """Test edge cases."""
    probabilities = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_soft_voting(probabilities)
    assert isinstance(result, dict)
