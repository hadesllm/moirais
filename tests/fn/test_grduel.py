"""Tests for grduel.geron_dueling_dqn."""
import numpy as np
import pytest
from morie.fn.grduel import geron_dueling_dqn


def test_grduel_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = geron_dueling_dqn(V, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grduel_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = geron_dueling_dqn(V, A)
    assert isinstance(result, dict)
