"""Tests for ali.ali_mikhail_haq_copula."""
import numpy as np
import pytest
from morie.fn.ali import ali_mikhail_haq_copula


def test_ali_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    result = ali_mikhail_haq_copula(y, u, v, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ali_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    result = ali_mikhail_haq_copula(y, u, v, theta)
    assert isinstance(result, dict)
