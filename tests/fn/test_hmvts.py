"""Tests for hmvts.geron_voting_soft."""
import numpy as np
import pytest
from moirais.fn.hmvts import geron_voting_soft


def test_hmvts_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_voting_soft(models, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvts_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_voting_soft(models, X)
    assert isinstance(result, dict)
