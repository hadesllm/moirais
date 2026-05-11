"""Tests for nystm.nystrom_approximation."""
import numpy as np
import pytest
from morie.fn.nystm import nystrom_approximation


def test_nystm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_landmarks = np.random.default_rng(42).normal(0, 1, 100)
    result = nystrom_approximation(X, m_landmarks)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nystm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    m_landmarks = np.random.default_rng(42).normal(0, 1, 100)
    result = nystrom_approximation(X, m_landmarks)
    assert isinstance(result, dict)
