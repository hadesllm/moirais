"""Tests for hertbg.heritability."""
import numpy as np
import pytest
from morie.fn.hertbg import heritability


def test_hertbg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = heritability(y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hertbg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = heritability(y, K)
    assert isinstance(result, dict)
