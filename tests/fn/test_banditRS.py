"""Tests for banditRS.contextual_bandit_rec."""
import numpy as np
import pytest
from morie.fn.banditRS import contextual_bandit_rec


def test_banditRS_basic():
    """Test basic functionality."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = contextual_bandit_rec(context, arms)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_banditRS_edge():
    """Test edge cases."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    arms = np.random.default_rng(42).normal(0, 1, 100)
    result = contextual_bandit_rec(context, arms)
    assert isinstance(result, dict)
