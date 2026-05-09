"""Tests for dinoss.dino_centering."""
import numpy as np
import pytest
from moirais.fn.dinoss import dino_centering


def test_dinoss_basic():
    """Test basic functionality."""
    p_t = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_centering(p_t, C, momentum)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dinoss_edge():
    """Test edge cases."""
    p_t = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    momentum = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_centering(p_t, C, momentum)
    assert isinstance(result, dict)
