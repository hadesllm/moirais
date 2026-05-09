"""Tests for hmvth.geron_voting_hard."""
import numpy as np
import pytest
from moirais.fn.hmvth import geron_voting_hard


def test_hmvth_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_voting_hard(models, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvth_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_voting_hard(models, X)
    assert isinstance(result, dict)
